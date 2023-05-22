import torch
import random

def logits_top_k(logits, filter_ratio = 0.5, minimum=1, pad_value=None):
    logits = logits.contiguous()
    if filter_ratio < 0:
        filter_ratio = - filter_ratio
    if filter_ratio >= 0 and filter_ratio <= 1.0: 
        num_logits = logits.shape[-1]
        k = max(int((1 - filter_ratio) * num_logits), minimum)
    else:
        k = max(int(filter_ratio), minimum)
    val, ind = torch.topk(input=logits, k=k, dim=-1)
    if pad_value is None:
        pad_value = float('-inf')
    probs = torch.full_like(logits, pad_value)
    # probs.scatter_(1, ind, val)
    probs.scatter_(-1, ind, val)
    return probs


def mask_with_top_k(x, k, largest=True, abs=True, pad_value=None):
    """
    mask the input tensor along the last dimension.
    The values the not in the topk will be masked as zeros
    
    """
    if abs:
        x_ = x.abs()
    else:
        x_ = x
    _, top_k_index = x_.topk(k=k, dim=-1, largest=largest) # BHW x K

    mask = torch.zeros_like(x)
    ones = torch.ones_like(x)
    mask.scatter_(-1, index=top_k_index, src=ones)

    x = x * mask
    if pad_value is None or pad_value != 0:
        if pad_value is None:
            pad_value = float('-inf')
        x[mask == 0] = x[mask == 0] + pad_value
    return x


def sample_index_randomly(x, k, filter_ratio=0, largest=True):
    """
    x: should be 2D tensor, randomly smaple along the lat dimension
    """
    assert x.dim() == 2, 'currently only two dimensional tensors are supprted!'
    
    if filter_ratio < 0:
        filter_ratio = - filter_ratio
    if filter_ratio >= 0 and filter_ratio <= 1.0: 
        num_logits = x.shape[-1]
        topk = max(int((1 - filter_ratio) * num_logits), k)
    else:
        topk = max(int(filter_ratio), k)
    
    _, top_k_index = x.topk(k=topk, dim=-1, largest=largest) # BHW x K

    sampled = []
    for i in range(x.shape[0]):
        index = top_k_index[i]
        sampled_ = torch.tensor(random.sample(index.tolist(), k)).to(index)
        sampled.append(sampled_)
    sampled = torch.stack(sampled, dim=0).to(top_k_index)
    return sampled

def get_token_type(mask, token_shape):
    """
    Get the token type according to the given mask and token_shape.
    Note that we treat tokens into 3 types.
    0: masked tokens
    1: unmasked tokens
    2: partially masked tokens   

    Args:
    mask: 4D tensor, B x 1 x H x W, the mask of the origin image. 1 denotes masked pixles 
    and 0 denotes unmasked pixels.
    token_shape: [H/r, W/r]. the shape of token
    """
    mask_float = mask.float()

    mask_unshuffle = pixel_unshuffle(mask_float, token_shape) # B x r^2 x H/r x W/r

    scale_factor = mask_unshuffle.shape[1]
    mask_unshuffle = mask_unshuffle.sum(dim=1, keepdim=True) # B x 1 x H/r x W/r

    token_type = torch.zeros_like(mask_unshuffle).long() + 2
                                                                    
    token_type[mask_unshuffle==0] = 0 # unmasked tokens
    token_type[mask_unshuffle==scale_factor] = 1 # fully masked tokens
    return token_type

def gen_attention_mask(H, W, type='full', causal=True, condition_seq_len=0, **kwargs):


    content_seq_len = H * W
    seq_len = content_seq_len + condition_seq_len
    mask = torch.zeros(seq_len, seq_len)

    mask[:, :condition_seq_len] = 1

    if type == 'full':
        mask += 1
    elif type == 'dalle_row':
        for idx in range(content_seq_len):
            h = idx // W
            w = idx % W
            for w_ in range(w-W, w+1):
                i = h * W + w_
                mask[idx+condition_seq_len][i+condition_seq_len] = 1

    elif type == 'dalle_col':
        for idx in range(content_seq_len):
            h = idx // W
            w = idx % W
            for h_ in range(h+1):
                i = h_ * W + w 
                mask[idx+condition_seq_len][i+condition_seq_len] = 1
    elif type == 'dalle_conv':
        kernel_size = kwargs['kernel_size']
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        k_h, k_w = kernel_size[0], kernel_size[1]
        half_k_h = int(k_h/2)
        half_k_w = int(k_w/2)
        step_over_w = W - k_w 
        
        for idx in range(content_seq_len):
            max_kernel_count = (half_k_h+1) * k_w 
            step_over_count = step_over_w * (half_k_h+1)

            max_pre = max_kernel_count + step_over_count
            max_pre = min(idx+1, max_pre)

            for i in range(max_pre):
                valid = False 
                a = i % W 
                if a > half_k_w and a <= half_k_w + step_over_w:
                    valid = False  
                else:
                    valid = True 
                if valid:
                    mask[idx+condition_seq_len][idx-i+condition_seq_len] = 1
    else:
        raise NotImplementedError('attention type {} not implemented!'.format(type))

    if causal:
        causal_mask = torch.tril(torch.ones(content_seq_len+condition_seq_len, content_seq_len+condition_seq_len))
        mask *= causal_mask
    
    return mask
