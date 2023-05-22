import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from getmusic.utils.misc import instantiate_from_config
import numpy as np
from torch.cuda.amp import autocast
import getmusic.utils.midi_config as mc

eps = 1e-8

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a): # log(1-e_a)
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b): # M + log(e_(a-M)+e_(b-M))
    maximum = torch.max(a, b) # e(-70) is about 0
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt = (1-at-ct)/N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    return at, bt, ct, att, btt, ctt

class DiffusionRFM(nn.Module):
    def __init__(
        self,
        *,
        roformer_config=None,
        diffusion_step=100,
        alpha_init_type='cos',
        auxiliary_loss_weight=0,
        adaptive_auxiliary_loss=False,
        mask_weight=[1,1],
    ):
        super().__init__()
        
        self.roformer = instantiate_from_config(roformer_config)
        self.amp = False
        self.num_classes = self.roformer.vocab_size + 1 # defined in vocabulary, add an additional mask
        self.cond_weight = self.roformer.cond_weight
        self.tracks = 14
        self.pad_index = mc.duration_max * mc.pos_resolution - 1
        self.figure_size = mc.bar_max * mc.beat_note_factor * mc.max_notes_per_bar * mc.pos_resolution
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss

        if alpha_init_type == "alpha1":
            at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1)
        else:
            print("alpha_init_type is Wrong !! ")

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)
        

        log_1_min_ct = log_1_min_a(log_ct) # log(1-e_a), log(1-ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct) # log(1-ctt)
        # M + log(e_(a-M)+e_(b-M))
        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        self.prior_ps = 1024   # max number to sample per step

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1) 
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def log_sample_categorical_infer(self, logits, figure_size):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = gumbel_noise + logits
        for i in range(self.tracks - 2): # do not decode chord
            track = sample[:,:, i * figure_size : (i+1) * figure_size]
            if i % 2 == 1: # duration 
                track[:, self.pad_index+1:-1, :] = -70 # only decode duration tokens
            else: # only decode pitch tokens in $i$-th track
                start = mc.tracks_start[i // 2]
                end = mc.tracks_end[i // 2]
                track[:,:self.pad_index, :] = -70 
                track[:,self.pad_index+1:start,:] = -70
                track[:,end+1:-1,:] = -70
            sample[:,:, i * figure_size : (i+1) * figure_size] = track
        sample = sample.argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t) # log q(xt|x0)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def predict_start(self, log_x_t, t, condition_pos):          # p(x0|xt)

        x_t = log_onehot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                out = self.roformer(x_t, t, condition_pos)
        else:
            out = self.roformer(x_t, t, condition_pos)

        log_pred = F.log_softmax(out.double(), dim=2).float()
        batch_size = log_x_t.size()[0]
        zero_vector = torch.zeros(batch_size, out.size()[1], 2).type_as(log_x_t)- 70

        log_pred = torch.cat((log_pred, zero_vector), dim=2)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred.transpose(2, 1)
    
    def q_posterior(self, log_x_start, log_x_t, t):
        x_len = log_x_t.size()[-1]
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t) # get sample
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) #select masked tokens
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t) # b,1,1
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, x_len) #[B, 1, content_seq_len]

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        
        # log(q(xt|xt_1,x0))
        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector

        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0) 
        
    def p_pred(self, log_x, t, condition_pos):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t, condition_pos)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t, condition_pos)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x, t, figure_size, condition_pos, not_empty_pos, sampled=None, to_sample=None):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob, log_x_recon = self.p_pred(log_x, t, condition_pos)

        max_sample_per_step = self.prior_ps  # max number to sample per step
        if t[0] > 0 and to_sample is not None:
            log_x_idx = log_onehot_to_index(log_x)

            # dim=1，vocaburay dimension
            score = torch.exp(log_x_recon).max(dim=1).values.softmax(dim=1)

            out = self.log_sample_categorical_infer(log_x_recon, figure_size)
            out_idx = log_onehot_to_index(out)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            
            # only content has score
            _score = torch.where(((1 - condition_pos) * not_empty_pos).type(torch.bool), _score, 0)
            # only mask has score
            _score[log_x_idx != self.num_classes - 1] = 0

            for j in range(6): # do not decode chord
                
                __score = _score[0][2*j * figure_size : (2*j+2) * figure_size]
                
                if __score.sum() == 0:
                    continue
                    
                n_sample = min(to_sample[j] - sampled[j], max_sample_per_step)

                if to_sample[j] - sampled[j] - n_sample == 1:
                    n_sample = to_sample[j] - sampled[j]
                if n_sample <= 0:
                    continue

                sel = torch.multinomial(__score, n_sample.item())
                
                out2_idx[0][2*j * figure_size : (2*j+2) * figure_size][sel] = out_idx[0][2*j * figure_size : (2*j+2) * figure_size][sel]
                
                sampled[j] += ((out2_idx[0][2*j * figure_size : (2*j+2) * figure_size] != self.num_classes - 1).sum() - (log_x_idx[0][2*j * figure_size : (2*j+2) * figure_size] != self.num_classes - 1).sum()).item()

            out = index_to_log_onehot(out2_idx, self.num_classes)
        else:
            # Gumbel sample
            out = self.log_sample_categorical_infer(model_log_prob, figure_size)
            sampled = None

        if to_sample is not None:
            return out, sampled
        else:
            return out

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True) 
            
            pt = pt_all.gather(dim=0, index=t) 
            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps 
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, condition_pos, not_empty_pos, is_train=True):

        assert x.size(2) == self.figure_size

        b, device = x.size(0), x.device

        condition_pos = condition_pos.view(b, -1)
        not_empty_pos = not_empty_pos.view(b, -1)
        
        x_start = x
        t, pt = self.sample_time(b, device, 'importance')

        log_x_start = index_to_log_onehot(x_start.view(b,-1).long(), self.num_classes) 

        log_xt = self.q_sample(log_x_start=log_x_start, t=t)

        log_empty = -70 * torch.ones_like(log_x_start)
        log_empty[:,-2,:] = 0

        # return condition to ground truth
        log_xt = torch.where(condition_pos.unsqueeze(1).type(torch.bool), log_x_start, log_xt)
        # empty
        log_xt = torch.where((1 - not_empty_pos).unsqueeze(1).type(torch.bool), log_empty, log_xt)

        log_x0_recon = self.predict_start(log_xt, t=t, condition_pos=condition_pos)
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start.view(b,-1).long()

        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)

        position_weight = not_empty_pos * ((1 - condition_pos) + self.cond_weight * condition_pos) # loss weight: condition/content = cond_weight / 1
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        
        L_t_1 = self.multinomial_kl(log_true_prob, log_model_prob)
        L_t_1 = L_t_1 * position_weight
        L_t_1 = sum_except_batch(L_t_1)
        
        L_0 = -log_categorical(log_x_start, log_model_prob)
        L_0 *=  position_weight
        L_0 = sum_except_batch(L_0)
        
        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * L_0 + (1. - mask) * L_t_1

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        loss1 = kl_loss / pt
        vb_loss = loss1
    
        if self.auxiliary_loss_weight != 0 and is_train==True:
            kl_aux = self.multinomial_kl(log_x_start[:,:-2,:], log_x0_recon[:,:-2,:])
            kl_aux = kl_aux * position_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * L_0 + (1. - mask) * kl_aux
            
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
 
            vb_loss += loss2
        
        scale = (position_weight != 0.0).sum(-1)

        scale = torch.where(scale == 0.0, 0.0, (1 / scale).double())

        return log_model_prob, vb_loss * scale, L_0 * scale

    def device(self):
        return self.roformer.input_transformers.encoder.embed_positions.weight.device

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeunet: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.roformer.named_parameters()}# if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(
            self, 
            x,
            tempo,
            condition_pos,
            not_empty_pos,
            return_loss=False, 
            return_logits=True, 
            return_att_weight=False,
            is_train=True,
            **kwargs):
        if kwargs.get('autocast') == True:
            self.amp = True

        if is_train == True:
            log_model_prob, loss, decoder_nll = self._train_loss(x, condition_pos, not_empty_pos)
            loss = loss.sum()
            decoder_nll = decoder_nll.sum()

        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['loss'] = loss 
            out['decoder_nll'] = decoder_nll
            
        self.amp = False
        return out

    def sample(
            self,
            x,
            tempo,
            not_empty_pos,
            condition_pos,
            skip_step=0,
            **kwargs):

        batch_size = x.size()[0]
        assert batch_size == 1
          
        num_to_be_generated_per_track = (x == self.num_classes - 1).sum(-1)[0]
        num_to_be_generated_per_track = num_to_be_generated_per_track[0::2] + num_to_be_generated_per_track[1::2]

        x = x.view(batch_size, -1).long()
        
        condition_pos = condition_pos.view(batch_size, -1)
        not_empty_pos = not_empty_pos.view(batch_size, -1)

        sample_len = x.size()[1]
        assert sample_len % self.tracks == 0
        figure_size = sample_len // self.tracks

        device = self.log_at.device
        start_step = self.num_timesteps

        log_x_start = index_to_log_onehot(x, self.num_classes)
        
        log_pad = torch.ones_like(log_x_start) * -70
        log_pad[:, self.pad_index, :] = 0
        log_empty = -70 * torch.ones_like(log_x_start)
        log_empty[:,-2,:] = 0

        log_p_x_on_y = -70 * torch.ones((batch_size, self.num_classes, sample_len),device=device)
        log_p_x_on_y[:,-1,:] = 0
        
        start_step = self.num_timesteps

        to_be_sampled_per_step = torch.floor(num_to_be_generated_per_track / (self.num_timesteps - 1))
        
        self.n_sample = to_be_sampled_per_step.unsqueeze(0).repeat(self.num_timesteps - 1, 1)

        compensate = num_to_be_generated_per_track - self.n_sample.sum(0)

        for i in range(6):
            while compensate[i] > 0:
                compensate_freq = torch.ceil(self.num_timesteps / compensate[i]).long().item()
                if compensate_freq > 1:
                    self.n_sample[0::compensate_freq, i] += 1
                compensate[i] = num_to_be_generated_per_track[i] - self.n_sample.sum(0)[i]
        
        self.n_sample = torch.cat([torch.ones(1, 7, device=self.n_sample.device), self.n_sample], dim=0)
        self.n_sample = torch.where(self.n_sample == 0, -1, self.n_sample).long()

        with torch.no_grad():
            for diffusion_index in tqdm(range(start_step-1, -1, -1)):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                sampled = torch.zeros(7, dtype=torch.long, device=self.n_sample.device)

                while (sampled < self.n_sample[diffusion_index]).any():
                    log_p_x_on_y = torch.where(condition_pos.unsqueeze(1).type(torch.bool), log_x_start, log_p_x_on_y)
                    log_p_x_on_y = torch.where((1-not_empty_pos).unsqueeze(1).type(torch.bool), log_empty, log_p_x_on_y)
                    log_p_x_on_y, sampled = self.p_sample(log_p_x_on_y, t, figure_size, condition_pos, not_empty_pos, sampled, self.n_sample[diffusion_index])     # log_z is log_onehot
                    if sampled is None:
                        assert t[0] == 0
                        break
                    
            log_p_x_on_y = torch.where(condition_pos.unsqueeze(1).type(torch.bool), log_x_start, log_p_x_on_y)
            log_p_x_on_y = torch.where((1-not_empty_pos).unsqueeze(1).type(torch.bool), log_pad, log_p_x_on_y)
            content_token = log_onehot_to_index(log_p_x_on_y) 
        
        return content_token.view(batch_size, self.tracks, -1)
