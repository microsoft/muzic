import os
import glob
import logging

import torch

logger = logging.getLogger(__name__)

segment_arange_cuda_module = None


def load_cuda_module():
    global segment_arange_cuda_module

    if segment_arange_cuda_module is not None:
        return segment_arange_cuda_module

    path = os.path.dirname(os.path.abspath(__file__))
    cpps = glob.glob(os.path.join(path, "cuda_src/*.cpp"))
    cudas = glob.glob(os.path.join(path, "cuda_src/*.cu"))
    sources = list(cpps) + list(cudas)

    from torch.utils.cpp_extension import load
    module = load(name='segment_arange_cuda',
                  sources=sources,
                  extra_cflags=['-O2'],
                  with_cuda=True,
                  verbose=False)

    segment_arange_cuda_module = module

    return segment_arange_cuda_module


def segment_arange(
    ranges: torch.Tensor,
    start: int,
    seq_len: int,
    pad_value,
    dtype=torch.long,
    no_cuda_kernel: bool = False
):
    """
    Fill a set of values into some ranges of 1D vector.
    Note: The ranges cannot be overlapped, or some unexpected behavior may appear. Current code does not check it.
    :param ranges: (num_ranges, 2)  begin, end
    :param start: int
    :param seq_len:
    :param pad_value:
    :param dtype:
    :param no_cuda_kernel:
    :return:
    """
    # Todo: Verify the effect of the segment_arange kernel.
    num_ranges, dim = ranges.shape
    assert dim == 2
    assert ranges[:, 0].le(ranges[:, 1]).all()
    if ranges.is_cuda and not no_cuda_kernel:
        return segment_arange_cuda(ranges, start, seq_len, pad_value, dtype=dtype)
    else:
        return segment_arange_pytorch(ranges, start, seq_len, pad_value, dtype=dtype)


def segment_arange_cuda(ranges, start, seq_len, pad_value, dtype=torch.long):
    if dtype not in (torch.long,):
        raise NotImplementedError
    num_ranges = ranges.shape[0]
    out = torch.full((seq_len,), pad_value, dtype=dtype, device=ranges.device)
    if num_ranges == 0:
        return out
    module = load_cuda_module()
    module.cuda_forward(ranges, start, num_ranges, out)
    return out


def segment_arange_pytorch(ranges, start, seq_len, pad_value, dtype=torch.long):
    out = torch.full((seq_len,), pad_value, dtype=dtype, device=ranges.device)
    for idx, (begin, end) in enumerate(ranges):
        out[begin: end] = torch.arange(start, start + (end - begin), dtype=dtype, device=ranges.device)
    return out
