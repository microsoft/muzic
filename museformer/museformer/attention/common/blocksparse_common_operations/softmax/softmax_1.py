import torch
from .....blocksparse import BlocksparseSoftmax


def do_attn_softmax_for_part(self, attn_scores, attn_mask, real_part, mask_again=False):
    attn_scores_for_real_part = attn_scores[real_part] \
        # samples list of (heads, head_selected_blocks, block, block)
    attn_mask = attn_mask[real_part]  # samples list of layout, block_mask
    bsz = len(attn_mask)
    result = [None] * bsz
    for sample_idx in range(bsz):
        sample_attn_mask = attn_mask[sample_idx]
        if sample_attn_mask is None:
            continue
        sample_layout, sample_block_mask = sample_attn_mask
        if sample_layout is None:
            continue

        if sample_block_mask.dtype == torch.uint8:
            sample_block_mask = sample_block_mask.eq(1)
        assert sample_block_mask.dtype == torch.bool
        result[sample_idx] = attn_scores_for_real_part[sample_idx].masked_fill(sample_block_mask[None], -10000)

        softmax_label = (real_part, 'softmax', self.layer_sv, sample_idx)
        if softmax_label in self.instant_pocket:
            softmax = self.instant_pocket[softmax_label]
        else:
            softmax = BlocksparseSoftmax(sample_layout, self.block_size)
            self.instant_pocket[softmax_label] = softmax

        temp = softmax(result[sample_idx])
        if mask_again:
            temp = temp.masked_fill(sample_block_mask[None], 0.0)
        if self.dropout_module is not None:
            temp = self.dropout_module(temp)
        result[sample_idx] = temp

    result = {real_part: result}
    return result
