import torch
from .....blocksparse import BlocksparseMatMul


def do_sample_av_mul_base(self, sample_attn_weights, sample_v, sample_layout, real_part, sample_idx, tgt_len):
    if sample_layout is None:
        _, head, head_dim = sample_v.shape
        return sample_v.new_zeros(tgt_len, 1, head, head_dim)
    sample_v = sample_v.transpose(0, 1)[:, None]  # (head, 1, reg_len, head_dim)
    dsd_matmul_key = (real_part, 'dsd_matmul', self.layer_sv, sample_idx)
    if dsd_matmul_key in self.instant_pocket:
        dsd_matmul = self.instant_pocket[dsd_matmul_key]
    else:
        dsd_matmul = BlocksparseMatMul(sample_layout, self.block_size, 'dsd',
                                       device=sample_v.device)
        self.instant_pocket[dsd_matmul_key] = dsd_matmul

    sample_out = dsd_matmul(sample_attn_weights, sample_v)  # (head, 1, tgt_len, head_dim)
    sample_out = sample_out.permute(2, 1, 0, 3)  # (tgt_len, 1, head, head_dim)

    return sample_out


def do_av_mul_for_part(self, attn_weights_inc_part, v, attn_mask, real_part, tgt_len):
    attn_weights_for_part = attn_weights_inc_part[real_part]
    # samples list of (head, head_selected_blocks, block, block)
    bsz = len(attn_weights_for_part)
    attn_mask = attn_mask[real_part]
    result = []
    for sample_idx in range(bsz):
        sample_v = v[:, sample_idx]
        sample_attn_weights = attn_weights_for_part[sample_idx]  # (head, head_selected_blocks, block, block)
        sample_layout = attn_mask[sample_idx][0]

        sample_out = do_sample_av_mul_base(self, sample_attn_weights, sample_v, sample_layout, real_part,
                                           sample_idx, tgt_len)
        result.append(sample_out)

    if bsz > 1:
        result = torch.cat(result, dim=1)  # (tgt_len, bsz, num_heads, head_dim)
    else:
        result = result[0].contiguous()

    return result
