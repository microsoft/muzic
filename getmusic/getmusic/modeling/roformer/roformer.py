import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import RoFormerPreTrainedModel, PretrainedConfig
from transformers import RoFormerModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from einops import rearrange

class RoFormerConfig(PretrainedConfig):

    model_type = "roformer"

    def __init__(
        self,
        vocab_size=None,
        embedding_size=96,
        tracks=14,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1536,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=None,
        rotary_value=False,
        use_cache=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.tracks = tracks
        self.embedding_size = hidden_size if embedding_size is None else embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rotary_value = rotary_value
        self.use_cache = use_cache

def timestep_embedding(timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class DiffusionRoFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.time_embeddings = nn.Sequential(
            nn.Linear(128, config.embedding_size),
            SiLU(),
            nn.Linear(config.embedding_size, config.embedding_size),
        )
        
        self.LayerNorm = nn.LayerNorm(config.embedding_size * config.tracks, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, timesteps=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        inputs_embeds = self.word_embeddings(input_ids) # b, t, l, d

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)
        
        time_embeddings = self.time_embeddings(timestep_embedding(timesteps, 128)).unsqueeze(1).unsqueeze(1) # b, 1, 1, d
        token_type_embeddings = self.token_type_embeddings(token_type_ids.long()) # b, t, l, d

        embeddings = inputs_embeds + token_type_embeddings + time_embeddings
        embeddings = rearrange(embeddings, 'b t l d -> b l (t d)')
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class DiffusionRoFormerModel(RoFormerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = DiffusionRoFormerEmbeddings(config)

        self.embeddings_project = nn.Sequential(
                                nn.Linear(config.embedding_size * config.tracks, config.hidden_size),
                                SiLU(),
                                nn.Linear(config.hidden_size, config.hidden_size),
                                )

        self.outputs_project = nn.Sequential(
                                nn.Linear(config.hidden_size, config.embedding_size * config.tracks),
                                SiLU(),
                                nn.Linear(config.embedding_size * config.tracks, config.embedding_size * config.tracks),
                                )

        self.encoder = RoFormerModel(config).encoder
        
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.embeddings.word_embeddings.weight
    
    def get_extended_attention_mask(self, attention_mask, input_shape, device):

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(
        self,
        input_ids,
        timesteps,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple[torch.Tensor]]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, track_num, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device=input_ids.device)

        encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, timesteps=timesteps
        )

        embedding_output = self.embeddings_project(embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        
        sequence_output = self.outputs_project(sequence_output)
        sequence_output = rearrange(sequence_output, 'b l (t d) -> b (t l) d', d=self.config.embedding_size)

        return self.lm_head(sequence_output)[:,:,:-2]
