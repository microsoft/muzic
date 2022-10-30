import torch

from .rpe_self_attention_v2s1 import RpeSelfAttentionV2S1
from ..common.blocksparse_common_operations.qk_mul.qk_mul_1 import do_qk_scores_for_part
from ..common.blocksparse_common_operations.softmax.softmax_1 import do_attn_softmax_for_part
from ..common.blocksparse_common_operations.av_mul.av_mul_1 import do_av_mul_for_part


class BlocksparseRpeSelfAttentionV2S1(RpeSelfAttentionV2S1):
    def __init__(self, *args, block_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

        # --- for indexing relative position embeddings ---
        # num_heads_arange = torch.arange(self.num_heads)[:, None, None, None]  # (num_heads, 1, 1, 1)
        # self.register_buffer('num_heads_arange', num_heads_arange, persistent=False)
        # block_size_arange = torch.arange(self.block_size)[None, None, :, None]  # (1, 1, block_size, 1)
        # self.register_buffer('block_size_arange', block_size_arange, persistent=False)

    # ============= Interfaces =============
    def do_qk_scores_for_sx(self, base_sum_q, base_sum_k, base_reg_k, bsz, sum_len, reg_len, attn_mask=None):
        # base_sum_q: (sum_len, bsz, heads, head_dim)
        # base_sum_k: (sum_len, bsz, heads, head_dim)
        # base_reg_k: (reg_len, bsz, heads, head_dim)
        tgt = base_sum_q
        src = torch.cat((base_sum_k, base_reg_k), dim=0)
        return do_qk_scores_for_part(self, tgt, src, bsz, sum_len, sum_len + reg_len, attn_mask, 'sx')

    def do_masking_for_sx(self, attn_scores_inc, attn_mask):
        return attn_scores_inc

    def do_attn_softmax_for_sx(self, attn_scores_inc_sr, attn_mask=None):
        return do_attn_softmax_for_part(self, attn_scores_inc_sr, attn_mask, 'sx')

    def do_av_mul_for_sx(self, attn_weights_inc_sr, base_sum_v, base_reg_v, attn_mask=None, tgt_len=None):
        v = torch.cat((base_sum_v, base_reg_v), dim=0)
        return do_av_mul_for_part(self, attn_weights_inc_sr, v, attn_mask, 'sx', tgt_len)

    def do_qk_scores_for_rx(
        self,
        reg_q, sum_k, reg_k,
        bsz, sum_len, reg_len,
        attn_mask=None
    ):
        if sum_k is None:
            k = reg_k
        else:
            k = torch.cat((sum_k, reg_k), dim=0)
        return do_qk_scores_for_part(self, reg_q, k, bsz, reg_len, sum_len + reg_len, attn_mask, 'rx')

    def do_masking_for_rx(self, attn_scores_inc, attn_mask):
        return attn_scores_inc

    def do_attn_softmax_for_rx(self, attn_scores_inc, attn_mask=None):
        return do_attn_softmax_for_part(self, attn_scores_inc, attn_mask, 'rx')

    def do_av_mul_for_rx(self, attn_weights_inc, base_sum_v2, base_reg_v, attn_mask=None, tgt_len=None):
        if base_sum_v2 is None:
            v = base_reg_v
        else:
            v = torch.cat((base_sum_v2, base_reg_v), dim=0)
        return do_av_mul_for_part(self, attn_weights_inc, v, attn_mask, 'rx', tgt_len)
