from .roformer import RoFormerConfig, DiffusionRoFormerModel
import torch.nn as nn
import torch

class DiffusionRoformerModel(nn.Module):

    def __init__(
        self,
        vocab_size=None,
        cond_weight=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        config = RoFormerConfig(vocab_size=vocab_size+1, pad_token_id=vocab_size-1)

        self.cond_weight = cond_weight

        self.input_transformers = DiffusionRoFormerModel(config)

    def forward(self, x, timesteps, condition_pos):
        
        b = x.size()[0]
        x_seq_len = x.size()[1]
        figure_size = x_seq_len // 14
        x = x.reshape(b, 14, figure_size)
        condition_pos = condition_pos.reshape(b, 14, figure_size)
      
        attention_mask = None
        # extrapolation
        if not self.training and figure_size > 512:
            e = torch.ones(b, figure_size, figure_size, device=x.device)
            attention_mask = torch.tril(e, 255) * torch.triu(e, -255)
            
        outputs = self.input_transformers(input_ids=x, token_type_ids=condition_pos, timesteps=timesteps, attention_mask=attention_mask)
    
        return outputs
