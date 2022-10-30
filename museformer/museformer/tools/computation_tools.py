from typing import Optional

import torch


def may_bi_op(
    op1,
    op2,
    x: (torch.Tensor, tuple),
    seq_len_1: int,
    seq_len_2: int,
    out_dim: Optional[int],
    as_tuple=False,
):
    """
    Do two different projections over two different parts of x.
    :param op1:
    :param op2:
    :param x: input, either tensor or tuple of two tensors
    :param seq_len_1: int
    :param seq_len_2: int
    :param out_dim: int
    :param as_tuple: bool
    :return:
    """
    if isinstance(x, torch.Tensor):
        seq_len, bsz, _ = x.shape
        assert seq_len == seq_len_1 + seq_len_2
        if id(op1) == id(op2):
            if as_tuple:
                r = op1(x)
                return r[:seq_len_1], r[seq_len_1:]
            return op1(x)
        x1 = x[:seq_len_1]
        x2 = x[seq_len_1:]
        assert x2.shape[0] == seq_len_2
    else:
        x1, x2 = x
        assert x1 is None or x1.shape[0] == seq_len_1
        assert x2 is None or x2.shape[0] == seq_len_2
        bsz = x2.shape[1]

    if as_tuple:
        return op1(x1) if op1 is not None and (x1 is not None and seq_len_1 > 0) else x1, \
               op2(x2) if op2 is not None and x2 is not None else x2
    out = x1.new_empty(seq_len_1 + seq_len_2, bsz, out_dim)
    if seq_len_1 > 0:
        out[:seq_len_1] = op1(x1) if op1 is not None else x1
    if seq_len_2 > 0:
        out[seq_len_1:] = op2(x2) if op2 is not None else x2
    return out


def bi_projection(
    proj1: torch.nn.Linear,
    proj2: torch.nn.Linear,
    x: (torch.Tensor, tuple),
    seq_len_1: int,
    seq_len_2: int,
    as_tuple=False,
):
    """
    Do two different projections over two different parts of x.
    :param proj1: Linear instance
    :param proj2: Linear instance
    :param x: input, either tensor or tuple of two tensors
    :param seq_len_1: int
    :param seq_len_2: int
    :param as_tuple:
    :return:
    """
    out_dim = proj2.weight.shape[0]
    return may_bi_op(proj1, proj2, x, seq_len_1, seq_len_2, out_dim, as_tuple=as_tuple)


def may_bi_add(x, add1, add2, seq_len_1, seq_len_2):
    if add1 is None and add2 is None:
        return x
    if isinstance(x, torch.Tensor):
        seq_len, bsz = x.shape[:2]
        assert seq_len == seq_len_1 + seq_len_2
        x1 = x[:seq_len_1]
        x2 = x[seq_len_1:]
        assert x2.shape[0] == seq_len_2
    else:
        x1, x2 = x
        assert x1.shape[0] == seq_len_1
        assert x2.shape[0] == seq_len_2
        bsz = x2.shape[1]
    out = x1.new_empty(seq_len_1 + seq_len_2, bsz, *x1.shape[2:])
    if add1 is None:
        out[:seq_len_1] = x1
    else:
        out[:seq_len_1] = x1 + add1
    if add2 is None:
        out[seq_len_1:] = x2
    else:
        out[seq_len_1:] = x2 + add2
    return out


def may_bi_cat(x1, x2, dim=0):
    if x1 is None or x1.shape[dim] == 0:
        return x2
    if x2 is None or x1.shape[dim] == 0:
        return x1
    return torch.cat((x1, x2), dim=dim)


def pad_embed_first_dim(x, pad_len):
    if pad_len is None or pad_len == 0:
        return x
    shape = x.shape
    first_dim = shape[0]
    new_shape = ((first_dim + pad_len,) + shape[1:])
    r = x.new_zeros(*new_shape)
    r[:first_dim] = x
    return r


def pad_2d(x, pad_len_1, pad_len_2, pad_value):
    if pad_len_1 is None:
        pad_len_1 = 0
    if pad_len_2 is None:
        pad_len_2 = 0
    if pad_len_1 == 0 and pad_len_2 == 0:
        return x
    bsz, len_1, len_2 = x.shape
    r = x.new_full((bsz, len_1 + pad_len_1, len_2 + pad_len_2), pad_value)
    r[:, :len_1, :len_2] = x
    return r


def projection_and_pad(proj: torch.nn.Linear, x: torch.Tensor, padded_len: int):
    """
    Do projection and pad to a specific length. Combine together to save memory.
    :param proj:
    :param x: (seq_len, bsz, in_dim)
    :param padded_len: target seq_len
    :return:
    """
    seq_len, bsz = x.shape[:2]
    if seq_len == padded_len:
        return proj(x)
    assert padded_len > seq_len
    out_dim = proj.weight.shape[0]
    out = x.new_zeros((padded_len, bsz, out_dim))
    out[:seq_len] = proj(x)
    return out


def transfer_chunk_points_to_ranges(chunk_points):
    """

    :param chunk_points: (bsz, max_chunk + 1)
    :return:
    """
    return torch.stack((chunk_points[:, :-1], chunk_points[:, 1:]), dim=-1)  # (bsz, max_chunk, 2)
