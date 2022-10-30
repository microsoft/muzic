import torch


def indexing_sample_rpe_base(self, sample_r, sample_r_indices, sample_layout, tgt_label, tgt_len, sample_idx):
    # sample_r: (heads, tgt_len, num_selected_distance)
    # sample_r_indices: (head_selected_blocks, block, block)
    # sample_layout: (1, tgt_block, src_block)
    sample_r = sample_r.view(self.num_heads, tgt_len // self.block_size, self.block_size, -1)
    sample_tgt_block_ids = sample_layout[0].nonzero()[:, 0]  # (head_selected_blocks,)

    temp_rpe = sample_r[
        self.num_heads_arange,  # (8, 1, 1, 1)
        sample_tgt_block_ids[None, :, None, None],  # (1, 4, 1, 1)
        self.block_size_arange,  # (1, 1, block_size, 1)
        sample_r_indices[None],  # (1, head_selected_blocks, block, block)
    ]  # (head, head_selected_blocks, block, block)

    return temp_rpe


def add_rpe_for_part(self, attn_scores, qs, r_list, rel_indices, bsz, query_len, attn_mask, part_label):
    attn_mask = attn_mask[part_label]  # samples list of tuple (layout, block_mask)
    attn_scores_for_part = attn_scores[part_label]  # samples list of (heads, head_selected_blocks, block, block)
    attn_scores_for_part_with_rpe = [item for item in attn_scores_for_part]
    for rel_idx in range(self.num_relation_types):
        r_indices = rel_indices[rel_idx]
        r_indices = r_indices[part_label]

        if r_indices is None:
            continue

        r_embed = r_list[rel_idx].view(-1, self.num_heads, self.head_dim) \
            # (num_selected_pos, heads, head_dim)
        r_qs = qs[rel_idx]  # (sum_len, bsz, heads, head_dim)
        temp_r = torch.einsum("ibhd,jhd->bhij", r_qs, r_embed)  # (bsz, heads, sum_len, num_selected_distance)
        for sample_idx in range(bsz):
            sample_r = temp_r[sample_idx]  # (heads, sum_len, num_selected_distance)
            sample_r_indices = r_indices[sample_idx]  # (head_selected_blocks, block, block)

            sample_layout = attn_mask[sample_idx][0]

            temp_rpe = indexing_sample_rpe_base(
                self, sample_r, sample_r_indices, sample_layout, part_label, query_len, sample_idx
            )

            attn_scores_for_part_with_rpe[sample_idx] = attn_scores_for_part_with_rpe[sample_idx] + temp_rpe

    attn_scores_for_part_with_rpe = {
        part_label: attn_scores_for_part_with_rpe
    }

    return attn_scores_for_part_with_rpe
