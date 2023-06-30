import random
from typing import List, Tuple, Union


def remove_instrument(remigen_tokens: List[str], instrument_token: str) -> List[str]:
    """
    Remove an instrument from the remigen sequence.
    Note that the empty onset is not deleted since it does not matter in most cases,
    and the tempo token may still carry valuable information.
    Args:
        remigen_tokens: remigen sequence
        instrument_token: the instrument token, e.g. "i-128"

    Returns:
        remigen sequence without the designated instrument

    """
    tokens = []

    # delete relevant part
    ignore_mode = False
    for token in remigen_tokens:
        if token.startswith('s'):
            raise NotImplementedError("This function is not implemented to support time signature token.")
        if ignore_mode:
            if any([token.startswith(pre) for pre in ('o', 't', 'b')]) or \
                    (token.startswith('i') and token != instrument_token):
                ignore_mode = False
        else:
            if token == instrument_token:
                ignore_mode = True

        if not ignore_mode:
            tokens.append(token)

    return tokens


def count_token_num(remigen_seq: List[str], token: str, return_indices=False) -> Union[int, Tuple[int, list]]:
    """
    Count the number of a specific token in a remigen sequence.
    Args:
        remigen_seq: remigen sequence
        token: token str
        return_indices: list, containing the indices of the tokens

    Returns:
        the number of the appearance of the token.
    """
    num = 0
    indices = []
    for idx, t in enumerate(remigen_seq):
        if token == t:
            num += 1
            indices.append(idx)

    if return_indices:
        return num, indices
    return num


def count_bar_num(
    remigen_seq: List[str], bar_token: str = 'b-1', return_bar_token_indices=False
) -> Union[Tuple[int, int], Tuple[int, List, int]]:
    """
    Count the number of bars, including the complete bars and a possible incomplete bar.
    Args:
        remigen_seq: remigen sequence
        bar_token: bar token string
        return_bar_token_indices: bool
    Returns:
        num_of_complete_bars: the number of complete bars.
        num_of_incomplete_bars: the number of incomplete bar (0 or 1).
                                If the sequence does not end with 'b-1', it is regarded as an incomplete bar.
    """
    result = count_token_num(remigen_seq, bar_token, return_indices=return_bar_token_indices)
    if remigen_seq[-1] != bar_token:
        num_incomplete_bar = 1
    else:
        num_incomplete_bar = 0
    if return_bar_token_indices:
        return result + (num_incomplete_bar,)
    return result, num_incomplete_bar


def get_bar_ranges(remigen_seq: List[str], bar_token: str = 'b-1'):
    _, bar_token_indices, num_incomplete_bar = count_bar_num(
        remigen_seq, bar_token=bar_token, return_bar_token_indices=True
    )
    
    complete_bar_result = []
    in_complete_bar_result = []

    begin = 0
    for end_index in bar_token_indices:
        complete_bar_result.append((begin, end_index + 1))
        begin = end_index + 1
    
    if num_incomplete_bar > 0:
        in_complete_bar_result.append((begin, len(remigen_seq)))
    
    return complete_bar_result, in_complete_bar_result


def get_instrument_played(remi_seq:List[str], max_num=None) -> List[str]:
    ret = set()
    for ev in remi_seq:
        if ev[0] == "i":
            ret.add(ev)
        if len(ret) == max_num:
            break
    ret = list(ret)
    return ret


def get_instrument_seq(remi_seq: List[str], instru_id: int) -> List[str]:
    chosen_events = []
    cur_pos = ""
    pos_tempo = []
    cur_instru = -1
    has_pushed = 0
    for i, note in enumerate(remi_seq):
        if note[0] == "o":
            cur_pos = note
            has_pushed = 0
        elif note[0] == 'b':
            chosen_events.append(note)
        elif note[0] == "t": # 在REMIGEN2里有bug，因为t只在bar的开头出现
            pos_tempo = [cur_pos, note]
            has_pushed = 0
        elif note[0] == "i":
            cur_instru = eval(note[2:])
        elif note[0] == "p":
            if cur_instru == instru_id:
                if not has_pushed:
                    chosen_events.extend(pos_tempo)
                    chosen_events.append(f"i-{instru_id}")
                    has_pushed = 1
                chosen_events.extend(remi_seq[i:i+3])
    return chosen_events


def sample_bars(input_seq: List[str], num_sampled_bars) -> Tuple[List[str], int]:
    assert num_sampled_bars > 0

    bar_ranges, incomplete_bar_range = get_bar_ranges(input_seq, bar_token='b-1')
    assert len(incomplete_bar_range) == 0
    num_bars = len(bar_ranges)
    assert num_bars > 0

    num_sampled_bars = min(num_bars, num_sampled_bars)

    sampled_begin = random.randint(0, num_bars - num_sampled_bars)
    start = bar_ranges[sampled_begin][0]
    end = bar_ranges[sampled_begin + num_sampled_bars - 1][1]
    return input_seq[start: end], num_sampled_bars
