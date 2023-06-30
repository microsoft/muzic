# Author: Botao Yu

from . import const
import inspect


def encoding_successive_cut(encoding,
                            bar_abbr,
                            max_length=None,
                            max_bar=None,
                            get_bar_offset=None,
                            authorize_right=None,
                            authorize_bar=None,
                            len_encoding=None,
                            max_bar_num=None,):
    assert inspect.isfunction(get_bar_offset)
    assert inspect.isfunction(authorize_right)
    assert inspect.isfunction(authorize_bar)
    if len_encoding is None:
        len_encoding = len(encoding)
    if max_length is None:
        max_length = len_encoding
    assert max_length > 0

    encodings = []

    start = 0
    while start < len_encoding:
        end = min(start + max_length, len_encoding)
        first_bar_idx, bar_offset = get_bar_offset(encoding[start: end])
        assert first_bar_idx == 0
        have_bar = False
        while True:
            assert end > start, "No authorized right position for the cut. " + \
                                ("However, there is a bar in the range." if have_bar
                                 else "And there is no bar in the range.")
            if end == len_encoding or authorize_right(encoding, end):
                have_bar = True
                if max_bar is None:
                    break
                else:
                    if authorize_bar(encoding, start, end, bar_offset, max_bar):
                        break
            end -= 1
        encodings.append(ensure_bar_idx(encoding[start: end], bar_offset, bar_abbr,
                                        max_bar_num=max_bar_num))
        start = end

    return encodings


def ensure_bar_idx(encoding, offset, bar_abbr, max_bar_num=None):
    new_encoding = []
    for item in encoding:
        if item[0] == bar_abbr:
            bar_idx = item[1]
            bar_idx -= offset
            if max_bar_num is not None and bar_idx >= max_bar_num:
                bar_idx = max_bar_num - 1
            new_encoding.append((bar_abbr, bar_idx))
        else:
            new_encoding.append(item)
    return new_encoding


def do_remove_bar_idx(encoding):
    new_encoding = []
    for item in encoding:
        if item[0] == const.BAR_ABBR:
            new_encoding.append((const.BAR_ABBR, 0))
        else:
            new_encoding.append(item)
    return new_encoding
