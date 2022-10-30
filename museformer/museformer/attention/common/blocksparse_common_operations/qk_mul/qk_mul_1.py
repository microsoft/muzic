from .....blocksparse import BlocksparseMatMul


def do_sample_qk_scores_base(
    self, sample_layout, sample_tgt, sample_src,
    tgt_len, src_len, tgt_label, sample_idx
):
    if sample_layout is None:
        return None

    # sample_layout: (1, tgt_block, src_block)
    assert sample_tgt.shape == (tgt_len, self.num_heads, self.head_dim)
    assert sample_src.shape == (src_len, self.num_heads, self.head_dim)
    assert sample_layout.shape == (1, tgt_len // self.block_size, src_len // self.block_size), \
        str(sample_layout.shape) + ' %d %d' % (tgt_len // self.block_size, src_len // self.block_size)

    if tgt_len == 0 or src_len == 0:
        return sample_tgt.new_empty(self.num_heads, 0, self.block_size, self.block_size)

    sdd_matmul_key = (tgt_label, 'sdd_matmul', self.layer_sv, sample_idx)
    if sdd_matmul_key in self.instant_pocket:
        sdd_matmul = self.instant_pocket[sdd_matmul_key]
    else:
        sdd_matmul = BlocksparseMatMul(sample_layout, self.block_size, 'sdd',
                                       device=sample_tgt.device, trans_b=True)
        self.instant_pocket[sdd_matmul_key] = sdd_matmul

    sample_attn_scores = sdd_matmul(
        sample_tgt.transpose(0, 1)[:, None],  # (heads, 1, sum_len, head_dim)
        sample_src.transpose(0, 1)[:, None],  # (heads, 1, reg_len, head_dim)
    )  # (heads, head_selected_blocks, block, block)
    assert sample_attn_scores.shape[1] == int(sample_layout[0].sum())

    return sample_attn_scores


def do_qk_scores_for_part(
    self,
    tgt, src,
    bsz, tgt_len, src_len,
    attn_mask, part_label,
):
    # tgt: (tgt_len, bsz, num_heads, head_dim)
    # src: (src_len, bsz, num_heads, head_dim)
    part_attn_mask = attn_mask[part_label]
    attn_scores = []
    for idx in range(bsz):
        sample_layout = part_attn_mask[idx][0]  # (1, tgt_block, src_block)
        sample_attn_scores = do_sample_qk_scores_base(
            self,
            sample_layout, tgt[:, idx], src[:, idx],
            tgt_len, src_len, part_label, idx
        )
        attn_scores.append(sample_attn_scores)
    attn_scores = {part_label: attn_scores}
    return attn_scores
