import torch
from torch import nn
from getmusic.utils.misc import instantiate_from_config

from torch.cuda.amp import autocast

def disabled_train(self, mode=True):
    return self


class DFM(nn.Module):
    def __init__(
        self,
        *,
        diffusion_config
    ):
        super().__init__()
        self.rfm = instantiate_from_config(diffusion_config)
        self.truncation_forward = False
    
    def forward(
        self,
        batch,
        name='none',
        **kwargs
    ):
        output = self.rfm(batch,  **kwargs)
        return output

    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                try: 
                    params += getattr(self, name).parameters(recurse=recurse, name=name)
                except:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    def device(self):
        return self.rfm.device()

    def get_ema_model(self):
        return self.rfm
    
    @torch.no_grad()
    def infer_sample(
        self,
        x,
        tempo,
        not_empty_pos,
        condition_pos,
        skip_step,
        **kwargs):

        self.eval()

        trans_out = self.rfm.sample(x=x,
                                        tempo=tempo,
                                        not_empty_pos=not_empty_pos,
                                        condition_pos=condition_pos,
                                        skip_step=skip_step,
                                        **kwargs)

        return trans_out
