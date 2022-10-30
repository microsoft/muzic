import torch
from fairseq.data import data_utils

from ..datasets.chunk_sequence_dataset_2 import get_bar_chunk_points


def construct_bar_chunk_info(reg_tokens, src_lengths, eob, begin_idx=0, only_bos_prefix=True, device='cpu'):
    """

    :param reg_tokens: (bsz, seq_len)
    :param src_lengths: (bsz,)
    :param eob:
    :param begin_idx:
    :param only_bos_prefix:
    :param device:
    :return:
    """
    assert only_bos_prefix
    chunk_points = []
    num_chunks = []
    should_minus_one = []
    bsz = len(reg_tokens)
    for idx, (sample_reg_tokens, sample_length) in enumerate(zip(reg_tokens, src_lengths)):
        sample_reg_tokens = sample_reg_tokens[:sample_length]
        sample_chunk_points, sample_is_complete_bar = get_bar_chunk_points(
            sample_reg_tokens,
            eob, begin_idx=begin_idx
        )  #
        chunk_points.append(sample_chunk_points)
        sample_num_chunks = len(sample_chunk_points) - 1
        num_chunks.append(sample_num_chunks)
        should_minus_one.append(not sample_is_complete_bar and sample_num_chunks > 0)
    chunk_points = data_utils.collate_tokens(
        chunk_points, 0
    ).to(device)  # (bsz, max_chunk + 1)
    num_chunks = torch.tensor(num_chunks, device=device)
    should_minus_one = torch.tensor(should_minus_one, device=device).long()
    num_complete_chunks = num_chunks - should_minus_one
    num_pref = torch.tensor((1,) * bsz, device=device)

    return chunk_points, num_chunks, num_complete_chunks, num_pref
