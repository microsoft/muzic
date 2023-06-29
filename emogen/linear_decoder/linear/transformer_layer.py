from typing import Dict, List, Optional

import torch
from torch import Tensor

from fairseq.modules.transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
# from fast_transformers.attention.attention_layer import AttentionLayer
# from fast_transformers.attention.causal_linear_attention import CausalLinearAttention
# from fast_transformers.masking import TriangularCausalMask, LengthMask, FullMask

from .causal_linear_attention import CausalLinearAttention
from .attention_layer import AttentionLayer
# from fast_transformers.transformers import TransformerEncoderLayer
from fast_transformers.attention import LinearAttention

class LinearTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__(args, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        self.decoder_attention_heads = args.decoder_attention_heads

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        causal_linear_attention = CausalLinearAttention(embed_dim)
        linear_attention_layer = AttentionLayer(causal_linear_attention,
                                                embed_dim, args.decoder_attention_heads)
        return linear_attention_layer

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # if prev_self_attn_state is not None:
        #     prev_key, prev_value = prev_self_attn_state[:2]
        #     saved_state: Dict[str, Optional[Tensor]] = {
        #         "prev_key": prev_key,
        #         "prev_value": prev_value,
        #     }
        #     if len(prev_self_attn_state) >= 3:
        #         saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
        #     assert incremental_state is not None
        #     self.self_attn._set_input_buffer(incremental_state, saved_state)
        # _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        # if self.cross_self_attention and not (
        #     incremental_state is not None
        #     and _self_attn_input_buffer is not None
        #     and "prev_key" in _self_attn_input_buffer
        # ):
        #     if self_attn_mask is not None:
        #         assert encoder_out is not None
        #         self_attn_mask = torch.cat(
        #             (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
        #         )
        #     if self_attn_padding_mask is not None:
        #         if encoder_padding_mask is None:
        #             assert encoder_out is not None
        #             encoder_padding_mask = self_attn_padding_mask.new_zeros(
        #                 encoder_out.size(1), encoder_out.size(0)
        #             )
        #         self_attn_padding_mask = torch.cat(
        #             (encoder_padding_mask, self_attn_padding_mask), dim=1
        #         )
        #     assert encoder_out is not None
        #     y = torch.cat((encoder_out, x), dim=0)
        # else:
        y = x

        x, attn = self.run_self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        assert self.encoder_attn is None and encoder_out is None
        # if self.encoder_attn is not None and encoder_out is not None:
        #     residual = x
        #     if self.normalize_before:
        #         x = self.encoder_attn_layer_norm(x)
        #     if prev_attn_state is not None:
        #         prev_key, prev_value = prev_attn_state[:2]
        #         saved_state: Dict[str, Optional[Tensor]] = {
        #             "prev_key": prev_key,
        #             "prev_value": prev_value,
        #         }
        #         if len(prev_attn_state) >= 3:
        #             saved_state["prev_key_padding_mask"] = prev_attn_state[2]
        #         assert incremental_state is not None
        #         self.encoder_attn._set_input_buffer(incremental_state, saved_state)
        #
        #     x, attn = self.encoder_attn(
        #         query=x,
        #         key=encoder_out,
        #         value=encoder_out,
        #         key_padding_mask=encoder_padding_mask,
        #         incremental_state=incremental_state,
        #         static_kv=True,
        #         need_weights=need_attn or (not self.training and self.need_attn),
        #         need_head_weights=need_head_weights,
        #     )
        #     x = self.dropout_module(x)
        #     x = self.residual_connection(x, residual)
        #     if not self.normalize_before:
        #         x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            raise NotImplementedError
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def run_self_attn(self, query, key_padding_mask, incremental_state, need_weights, **kwargs):
        if incremental_state is not None:
            raise NotImplementedError
        if need_weights:
            raise NotImplementedError

        # tgt_len, bsz, embed_dim = query.shape
        # src_len = tgt_len
        # num_heads = self.decoder_attention_heads
        # head_dim = self.embed_dim // num_heads

        # query = query.transpose(0, 1).reshape(bsz, tgt_len, num_heads, head_dim)
        query = query.transpose(0, 1)

        if key_padding_mask is not None:
            key_padding_mask = ~key_padding_mask

        r = self.self_attn(query, query, query, attn_mask=None, key_padding_mask=key_padding_mask)

        r = r.transpose(0, 1)

        return r, None

# class LinearTransformerEncoderLayer(TransformerEncoderLayer):

