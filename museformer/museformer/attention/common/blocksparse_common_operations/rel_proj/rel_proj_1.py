import torch
import torch.nn.functional as F
from .....kernels.range_fill import range_fill


def select_and_do_r_proj(self, rel_indices):
    """

    :param rel_indices: relation list of real_part dict
    :return:
    """

    r_embed_list = []
    r_modified_indices = []
    for rel_idx, one_rel_indices in enumerate(rel_indices):
        # Collect those indices for one kind of relative positions that are used for all the samples and real_parts
        one_rel_used_indices = []
        for real_part in one_rel_indices:  # ('sr', 'rx')
            one_rel_part_indices = one_rel_indices[real_part]
            if one_rel_part_indices is None:
                continue
            for sample_rel_indices in one_rel_part_indices:
                assert sample_rel_indices.shape[1:] == (self.block_size, self.block_size)
                sample_used_rel_indices = torch.unique(sample_rel_indices)  # (num_unique,)
                one_rel_used_indices.append(sample_used_rel_indices)
        one_rel_used_indices = torch.cat(one_rel_used_indices).unique()

        rel_selected_embed = F.embedding(one_rel_used_indices, self.rel_embeddings[rel_idx], padding_idx=0)

        rel_proj = getattr(self, 'rel%d_proj' % rel_idx, None)
        if rel_proj is not None:
            rel_selected_embed = rel_proj(rel_selected_embed)

        label_transform = range_fill(
            torch.stack((one_rel_used_indices, one_rel_used_indices + 1), dim=-1),
            torch.arange(len(one_rel_used_indices), device=one_rel_used_indices.device),
            self.num_rel_embeddings[rel_idx], 0
        )

        one_r_indices = {}
        for real_part in one_rel_indices:
            one_rel_part_indices = one_rel_indices[real_part]
            if one_rel_part_indices is None:
                one_r_indices[real_part] = None
                continue
            samples_r_indices = []
            for sample_rel_indices in one_rel_part_indices:
                sample_r_indices = label_transform[sample_rel_indices]
                samples_r_indices.append(sample_r_indices)
            one_r_indices[real_part] = samples_r_indices

        r_embed_list.append(rel_selected_embed)
        r_modified_indices.append(one_r_indices)

    return r_embed_list, r_modified_indices
