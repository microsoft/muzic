import os
import glob
import logging

import torch

logger = logging.getLogger(__name__)

block_fill_cuda_module = None


def load_cuda_module():
    global block_fill_cuda_module

    if block_fill_cuda_module is not None:
        return block_fill_cuda_module

    path = os.path.dirname(os.path.abspath(__file__))
    cpps = glob.glob(os.path.join(path, "cuda_src/*.cpp"))
    cudas = glob.glob(os.path.join(path, "cuda_src/*.cu"))
    sources = list(cpps) + list(cudas)

    from torch.utils.cpp_extension import load
    module = load(name='block_fill_cuda',
                  sources=sources,
                  extra_cflags=['-O2'],
                  with_cuda=True,
                  verbose=False)

    block_fill_cuda_module = module

    return block_fill_cuda_module


def block_fill_(out, block_ranges, head_masks, fill_value, no_cuda_kernel=False):
    """

    :param block_ranges: (num_blocks, 4)
    :param head_masks: (num_blocks, num_heads)
    :param out:
    :param fill_value:
    :param no_cuda_kernel:
    :return:
    """
    if block_ranges.shape[0] == 0:
        return out

    assert isinstance(fill_value, bool)
    if block_ranges.is_cuda and not no_cuda_kernel:
        return block_fill_cuda(out, block_ranges, head_masks, fill_value)
    else:
        return block_fill_pytorch(out, block_ranges, head_masks, fill_value)


def _check_input(block_ranges, head_masks, out):
    assert head_masks.dtype == torch.bool
    assert block_ranges.device == head_masks.device
    assert block_ranges.device == out.device
    num_blocks, col = block_ranges.shape
    num_blocks_2, num_heads = head_masks.shape
    assert num_blocks == num_blocks_2
    assert col == 4
    assert num_heads > 0
    return num_blocks, num_heads


def block_fill_cuda(out, block_ranges, head_masks, fill_value):
    num_blocks, num_heads = _check_input(block_ranges, head_masks, out)

    num_heads_2, seq_len_1, seq_len_2 = out.shape
    assert num_heads_2 == num_heads
    assert block_ranges.is_contiguous()
    assert head_masks.is_contiguous()
    assert out.is_contiguous()

    max_query = (block_ranges[:, 1] - block_ranges[:, 0]).max().item()
    max_key = (block_ranges[:, 3] - block_ranges[:, 2]).max().item()

    if max_query <= 0 or max_key <= 0:
        return out

    module = load_cuda_module()

    module.cuda_forward(block_ranges, head_masks, num_blocks, max_query, max_key, num_heads,
                        seq_len_1, seq_len_2, fill_value, out)

    return out


def block_fill_pytorch(out, block_ranges, head_masks, fill_value):
    num_blocks, num_heads = _check_input(block_ranges, head_masks, out)

    assert out.shape[0] == num_heads
    num_heads_2, _, _ = out.shape
    assert num_heads_2 == num_heads

    for (query_begin, query_end, key_begin, key_end), line_mask in zip(block_ranges, head_masks):
        line_head_idx = torch.nonzero(line_mask, as_tuple=False).squeeze(-1)  # (num,)
        out[line_head_idx, query_begin: query_end, key_begin: key_end] = fill_value

    return out
