import math

import torch
import torch.nn as nn
from fairseq.modules.fairseq_dropout import FairseqDropout

from ...data_structures.four_dim_pocket import FourDimPocket
from ..common import common_funcs


class RpeSelfAttentionV2S1(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,

        num_summary,

        layer_idx,

        dropout=0.0,  # attention dropout

        query_proj_bias=True,
        key_proj_bias=True,
        value_proj_bias=True,
        sum_key2_proj_bias=True,
        sum_value2_proj_bias=True,
        out_proj_bias=True,
        add_different_kqv_bias_for_sum_and_reg=False,
        add_different_out_bias_for_sum_and_reg=False,

        share_query_proj=False,
        share_key_proj=False,
        share_value_proj=False,
        share_out_proj=False,
        share_key2_value2_proj_weight=False,

        max_summary=None,

        no_sum_out=False,

        single_head_masks=False,

        **kwargs
    ):
        assert single_head_masks, "Currently, we only support single head masks."
        # common_funcs.print_redundant_params(kwargs, self.__class__.__name__)

        super().__init__()
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.single_head_masks = single_head_masks
        self.num_summary = num_summary
        self.max_summary = self.num_summary if max_summary is None else max_summary

        self.no_sum_out = no_sum_out

        self.pocket = FourDimPocket()
        self.instant_pocket = self.pocket['instant']
        constant_pocket = self.pocket['constant']
        layer_to_sv = constant_pocket['layer_to_sv']
        self.layer_sv = layer_to_sv[self.layer_idx]

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "attention_embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        if add_different_kqv_bias_for_sum_and_reg:
            query_proj_bias = False
            key_proj_bias = False
            value_proj_bias = False
            for proj_name in ('query', 'key', 'value'):
                for target in (('sum', 'reg') if self.num_summary > 0 else ('reg',)):
                    bias_tensor = torch.zeros(self.embed_dim)
                    self.register_parameter(
                        '%s_%s_bias' % (target, proj_name),
                        nn.Parameter(bias_tensor, requires_grad=True)
                    )

        if add_different_out_bias_for_sum_and_reg:
            out_proj_bias = False
            for target in (('sum', 'reg') if not self.no_sum_out and self.num_summary > 0 else ('reg',)):
                bias_tensor = torch.zeros(self.embed_dim)
                self.register_parameter(
                    '%s_out_bias' % target,
                    nn.Parameter(bias_tensor, requires_grad=True)
                )

        self.reg_query_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=query_proj_bias)
        self.reg_key_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=key_proj_bias)
        self.reg_value_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=value_proj_bias)
        self.reg_out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=out_proj_bias)

        if self.num_summary > 0:
            if share_query_proj:
                self.sum_query_proj = self.reg_query_proj
            else:
                self.sum_query_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=query_proj_bias)

            if share_key_proj:
                self.sum_key_proj = self.reg_key_proj
            else:
                self.sum_key_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=key_proj_bias)

            if share_value_proj:
                self.sum_value_proj = self.reg_value_proj
            else:
                self.sum_value_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=value_proj_bias)

            if not self.no_sum_out:
                if share_out_proj:
                    self.sum_out_proj = self.reg_out_proj
                else:
                    self.sum_out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=out_proj_bias)

            self.share_key2_value2_proj_weight = share_key2_value2_proj_weight
            if share_key2_value2_proj_weight:
                self.sum_key2_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
                self.sum_value2_proj = self.sum_key2_proj
                if sum_key2_proj_bias:
                    self.sum_key2_bias = nn.Parameter(torch.zeros(self.embed_dim), requires_grad=True)
                if sum_value2_proj_bias:
                    self.sum_value2_bias = nn.Parameter(torch.zeros(self.embed_dim), requires_grad=True)
            else:
                self.sum_key2_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=sum_key2_proj_bias)
                self.sum_value2_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=sum_value2_proj_bias)

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        ) if dropout > 0.0 else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.reg_query_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.reg_key_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.reg_value_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.reg_out_proj.weight)
        if self.num_summary > 0:
            if id(self.sum_query_proj) != id(self.reg_query_proj):
                nn.init.xavier_uniform_(self.sum_query_proj.weight, gain=1 / math.sqrt(2))
            if id(self.sum_key_proj) != id(self.reg_key_proj):
                nn.init.xavier_uniform_(self.sum_key_proj.weight, gain=1 / math.sqrt(2))
            if id(self.sum_value_proj) != id(self.reg_value_proj):
                nn.init.xavier_uniform_(self.sum_value_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.sum_key2_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.sum_value2_proj.weight, gain=1 / math.sqrt(2))
            if not self.no_sum_out and id(self.sum_out_proj) != id(self.reg_out_proj):
                nn.init.xavier_uniform_(self.sum_out_proj.weight)

        nn.init.xavier_uniform_(self.reg_out_proj.weight)
        if self.reg_out_proj.bias is not None:
            nn.init.constant_(self.reg_out_proj.bias, 0.0)

    def forward(
        self,
        x: tuple,  # (sum_len, bsz, embed_dim), (reg_len, bsz, embed_dim)
        sum_token_ids,  # (bsz, sum_len)
        sum_len,
        reg_len,
        key_padding_mask=None,  # (bsz, all_seq_len)
        attn_mask=None,
        need_weights: bool = False,
        need_head_weights: bool = False,
        *args, **kwargs,
    ):
        if key_padding_mask is not None:
            raise NotImplementedError("Please combine key_padding_mask into attn_mask ahead.")
        del key_padding_mask

        if need_head_weights:
            need_weights = True

        # ===== Input Checking =====
        sum_x, reg_x = x
        bsz = reg_x.shape[1]
        del sum_token_ids

        # ===== Summarize =====
        base_reg_k = self.reg_key_proj(reg_x)
        bias = getattr(self, 'reg_key_bias', None)
        if bias is not None:
            base_reg_k = base_reg_k + bias
        base_reg_k = base_reg_k.view(reg_len, bsz, self.num_heads, self.head_dim)
        # base_reg_k: (reg_len, bsz, num_heads, head_dim)

        base_reg_v = self.reg_value_proj(reg_x)
        bias = getattr(self, 'reg_value_bias', None)
        if bias is not None:
            base_reg_v = base_reg_v + bias
        base_reg_v = base_reg_v.view(reg_len, bsz, self.num_heads, self.head_dim)
        # base_reg_v: (reg_len, bsz, num_heads, head_dim)

        if sum_len > 0:
            base_sum_q = self.sum_query_proj(sum_x)
            bias = getattr(self, 'sum_query_bias', None)
            if bias is not None:
                base_sum_q = base_sum_q + bias
            base_sum_q = base_sum_q.view(sum_len, bsz, self.num_heads, self.head_dim)
            # base_sum_q: (sum_len, bsz, num_heads, head_dim)

            base_sum_k = self.sum_key_proj(sum_x)
            bias = getattr(self, 'sum_key_bias', None)
            if bias is not None:
                base_sum_k = base_sum_k + bias
            base_sum_k = base_sum_k.view(sum_len, bsz, self.num_heads, self.head_dim)

            base_sum_v = self.sum_value_proj(sum_x)
            bias = getattr(self, 'sum_value_bias', None)
            if bias is not None:
                base_sum_v = base_sum_v + bias
            base_sum_v = base_sum_v.view(sum_len, bsz, self.num_heads, self.head_dim)

            attn_scores_inc_sx = self.do_qk_scores_for_sx(
                base_sum_q, base_sum_k, base_reg_k,
                bsz, sum_len, reg_len,
                attn_mask=attn_mask
            )  # real_parts dict of sample list of (heads, head_selected_blocks, block, block)

            for real_part in attn_scores_inc_sx:
                for sample_item in attn_scores_inc_sx[real_part]:
                    sample_item.mul_(self.scaling)

            attn_scores_inc_sx = self.do_masking_for_sx(attn_scores_inc_sx, attn_mask)

            attn_weights_inc_sx = self.do_attn_softmax_for_sx(attn_scores_inc_sx, attn_mask=attn_mask)
            del attn_scores_inc_sx

            sum_x2 = self.do_av_mul_for_sx(
                attn_weights_inc_sx, base_sum_v, base_reg_v, attn_mask=attn_mask, tgt_len=sum_len
            )  # samples list of (sum_len, 1, num_heads, head_dim)
            assert sum_x2.shape == (sum_len, bsz, self.num_heads, self.head_dim)
            sum_x2 = sum_x2.view(sum_len, bsz, self.embed_dim)

            if self.share_key2_value2_proj_weight:
                base_sum_k2 = self.sum_key2_proj(sum_x2)
                base_sum_v2 = base_sum_k2
                sum_key2_bias = getattr(self, 'sum_key2_bias', None)
                if sum_key2_bias is not None:
                    base_sum_k2 = base_sum_k2 + sum_key2_bias
                sum_value2_bias = getattr(self, 'sum_value2_bias', None)
                if sum_value2_bias is not None:
                    base_sum_v2 = base_sum_v2 + sum_value2_bias
                base_sum_k2 = base_sum_k2.view(sum_len, bsz, self.num_heads, self.head_dim)
                base_sum_v2 = base_sum_v2.view(sum_len, bsz, self.num_heads, self.head_dim)
            else:
                base_sum_k2 = self.sum_key2_proj(sum_x2).view(sum_len, bsz, self.num_heads, self.head_dim)
                base_sum_v2 = self.sum_value2_proj(sum_x2).view(sum_len, bsz, self.num_heads, self.head_dim)

        else:
            sum_x2 = reg_x.new_empty(0, bsz, self.embed_dim)
            base_sum_k2 = None
            base_sum_v2 = None

        # ===== Updating =====
        base_reg_q = self.reg_query_proj(reg_x)
        reg_query_bias = getattr(self, 'reg_query_bias', None)
        if reg_query_bias is not None:
            base_reg_q = base_reg_q + reg_query_bias
        base_reg_q = base_reg_q.view(reg_len, bsz, self.num_heads, self.head_dim)

        attn_scores_inc_rx = self.do_qk_scores_for_rx(
            base_reg_q, base_sum_k2, base_reg_k,
            bsz, sum_len, reg_len, attn_mask=attn_mask,
        )

        for real_part in attn_scores_inc_rx:
            for sample_item in attn_scores_inc_rx[real_part]:
                sample_item.mul_(self.scaling)

        attn_scores_inc_rx = self.do_masking_for_rx(attn_scores_inc_rx, attn_mask)

        attn_weights_inc_rx = self.do_attn_softmax_for_rx(attn_scores_inc_rx, attn_mask=attn_mask)

        # if self.layer_idx == 3:
        #     with open('attn_weights_inc_rx.bin', 'wb') as f:
        #         torch.save(attn_weights_inc_rx, f)
        #         print('saved attn_weights_inc_rx')

        reg_output = self.do_av_mul_for_rx(
            attn_weights_inc_rx, base_sum_v2, base_reg_v, attn_mask=attn_mask, tgt_len=reg_len
        )  # (reg_len, bsz, num_heads, head_dim)

        # ----- gate to combine sum_output and reg_output -----
        reg_output = reg_output.view(reg_len, bsz, self.embed_dim)
        reg_output = self.reg_out_proj(reg_output)
        reg_out_bias = getattr(self, 'reg_out_bias', None)
        if reg_out_bias is not None:
            reg_output = reg_output + reg_out_bias
        if not self.no_sum_out and self.num_summary > 0:
            sum_output = self.sum_out_proj(sum_x2)
            sum_out_bias = getattr(self, 'sum_out_bias', None)
            if sum_out_bias is not None:
                sum_output = sum_output + sum_out_bias
        else:
            sum_output = None

        if need_weights:
            raise NotImplementedError
        else:
            attn_weights = None

        return (sum_output, reg_output), attn_weights
        # (sum_len, bsz, embed_dim)  (reg_len, bsz, embed_dim)
        # None, (bsz, num_heads, all_seq_len, all_seq_len) or (bsz, all_seq_len, all_seq_len)

    def do_qk_scores_for_sx(self, base_sum_q, base_sum_k, base_reg_k, bsz, sum_len, reg_len, **kwargs):
        raise NotImplementedError

    def do_masking_for_sx(self, attn_scores_inc, attn_mask):
        raise NotImplementedError

    def do_attn_softmax_for_sx(self, attn_scores_for_sum, **kwargs):
        raise NotImplementedError

    def do_av_mul_for_sx(self, attn_weights_inc_sr, base_sum_v, base_reg_v, **kwargs):
        raise NotImplementedError

    def do_qk_scores_for_rx(
        self,
        reg_q, sum_k, reg_k,
        bsz, sum_len, reg_len,
        **kwargs
    ):
        raise NotImplementedError

    def do_masking_for_rx(self, attn_scores_for_reg, attn_mask):
        raise NotImplementedError

    def do_attn_softmax_for_rx(self, attn_scores_for_reg, attn_mask=None):
        raise NotImplementedError

    def do_av_mul_for_rx(self, attn_weights_inc, base_sum_v2, base_reg_v, **kwargs):
        raise NotImplementedError
