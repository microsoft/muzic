from typing import List, Tuple
from copy import deepcopy
import msgpack


def get_bar_num_from_sample_tgt_pos(sample_tgt_pos: List[List[Tuple[List[int], List[int]]]]) -> int:
    """
    Get the number of bars for a sample.
    Args:
        sample_tgt_pos: The target pos information for a sample.
                        If the output of the above pos_preprocess func is denoted as pos,
                        then this argument is pos[sample_idx][1]

    Returns:

    """
    num = 0
    for _, seg_bars in sample_tgt_pos:
        num += len(seg_bars)
    return num


def fill_pos_ts_and_tempo_(pos_info):
    cur_ts = pos_info[0][1]
    cur_tempo = pos_info[0][3]
    assert cur_ts is not None
    assert cur_tempo is not None
    for idx in range(len(pos_info)):
        pos_item = pos_info[idx]
        if pos_item[1] is not None:
            cur_ts = pos_item[1]
        if pos_item[3] is not None:
            cur_tempo = pos_item[3]
        if pos_item[2] == 0:
            if pos_item[1] is None:
                pos_item[1] = cur_ts
            if pos_item[3] is None:
                pos_item[3] = cur_tempo
    return pos_info


def string_pos_info_inst_id_(pos_info):
    for pos_item in pos_info:
        insts_notes = pos_item[-1]
        if insts_notes is None:
            continue
        inst_ids = tuple(insts_notes.keys())
        for inst_id in inst_ids:
            insts_notes[str(inst_id)] = insts_notes.pop(inst_id)
    return pos_info


def string_pos_info_inst_id(pos_info):
    pos_info = deepcopy(pos_info)
    return string_pos_info_inst_id_(pos_info)


def destring_pos_info_inst_id_(pos_info):
    for pos_item in pos_info:
        insts_notes = pos_item[-1]
        if insts_notes is None:
            continue
        inst_ids = tuple(insts_notes.keys())
        for inst_id in inst_ids:
            insts_notes[int(inst_id)] = insts_notes.pop(inst_id)
    return pos_info


def destring_pos_info_inst_id(pos_info):
    pos_info = deepcopy(pos_info)
    return destring_pos_info_inst_id_(pos_info)


def serialize_pos_info(pos_info, need_string_inst_id=True):
    if need_string_inst_id:
        pos_info = string_pos_info_inst_id(pos_info)
    return msgpack.dumps(pos_info)


def deserialize_pos_info(pos_info, need_destring_inst_id=True):
    pos_info = msgpack.loads(pos_info)
    for pos_item in pos_info:
        ts = pos_item[1]
        if ts is not None:
            pos_item[1] = tuple(ts)
    if need_destring_inst_id:
        pos_info = destring_pos_info_inst_id_(pos_info)
    return pos_info
