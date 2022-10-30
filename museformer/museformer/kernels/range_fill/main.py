import os
import glob
import logging

import torch

logger = logging.getLogger(__name__)

range_fill_cuda_module = None


def load_cuda_module():
    global range_fill_cuda_module

    if range_fill_cuda_module is not None:
        return range_fill_cuda_module

    path = os.path.dirname(os.path.abspath(__file__))
    cpps = glob.glob(os.path.join(path, "cuda_src/*.cpp"))
    cudas = glob.glob(os.path.join(path, "cuda_src/*.cu"))
    sources = list(cpps) + list(cudas)

    from torch.utils.cpp_extension import load
    module = load(name='range_fill_cuda',
                  sources=sources,
                  extra_cflags=['-O2'],
                  with_cuda=True,
                  verbose=False)

    range_fill_cuda_module = module

    return range_fill_cuda_module


def range_fill(
    ranges: torch.Tensor,
    values: torch.Tensor,
    seq_len: int,
    pad_value,
    dtype=torch.long,
    no_cuda_kernel: bool = False
):
    """
    Fill a set of values into some ranges of 1D vector.
    Note: The ranges cannot be overlapped, or some unexpected behavior may appear. Current code does not check it.
    :param ranges: (num_ranges, 2)  begin, end
    :param values: (num_ranges,)
    :param seq_len:
    :param pad_value:
    :param dtype:
    :param no_cuda_kernel:
    :return:
    """
    num_ranges, dim = ranges.shape
    values_shape, = values.shape
    assert num_ranges == values_shape
    assert dim == 2
    assert ranges[:, 0].lt(ranges[:, 1]).all()
    if ranges.is_cuda and not no_cuda_kernel:
        return range_fill_cuda(ranges, values, seq_len, pad_value, dtype=dtype)
    else:
        return range_fill_pytorch(ranges, values, seq_len, pad_value, dtype=dtype)


def range_fill_cuda(ranges, values, seq_len, pad_value, dtype=torch.long):
    if dtype not in (torch.long,):
        raise NotImplementedError
    num_ranges = ranges.shape[0]
    out = torch.full((seq_len,), pad_value, dtype=dtype, device=values.device)
    if num_ranges == 0:
        return out
    module = load_cuda_module()
    module.cuda_forward(ranges, values, num_ranges, out)
    return out


def range_fill_pytorch(ranges, values, seq_len, pad_value, dtype=torch.long):
    out = torch.full((seq_len,), pad_value, dtype=dtype, device=values.device)
    for idx, (begin, end) in enumerate(ranges):
        out[begin: end] = values[idx]
    return out
