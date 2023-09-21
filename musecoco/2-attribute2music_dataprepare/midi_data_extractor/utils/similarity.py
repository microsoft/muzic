import os
from multiprocessing import Pool
from functools import partial
import numpy as np

from .data import convert_dict_key_to_str, convert_dict_key_to_int


def cal_bar_similarity_basic(bar1_insts_poses_notes, bar2_insts_poses_notes, bar1_insts, bar2_insts,
                             ignore_pitch=False, ignore_duration_for_drum=True):
    o_sim = {}
    for inst in ((bar1_insts - bar2_insts) | (bar2_insts - bar1_insts)):
        o_sim[inst] = 'O'
    for inst in (bar1_insts & bar2_insts):
        inst_bar1 = bar1_insts_poses_notes[inst]
        inst_bar2 = bar2_insts_poses_notes[inst]
        # inst_bar1_pos = set(inst_bar1.keys())
        # inst_bar2_pos = set(inst_bar2.keys())
        # num_union_pos = len(inst_bar1_pos | inst_bar2_pos)
        # inter_pos = inst_bar1_pos & inst_bar2_pos

        if inst == 128:
            if ignore_pitch and ignore_duration_for_drum:
                raise ValueError("only_duration and ignore_duration_for_drum cannot be True at the same time for drum.")

        inst_bar1_note = set()
        for pos in inst_bar1:
            temp_pos_notes = inst_bar1[pos]
            for note in temp_pos_notes:
                info_tuple = [pos]
                if not ignore_pitch:
                    info_tuple.append(note[0])
                if not ignore_duration_for_drum or inst != 128:
                    info_tuple.append(note[1])
                info_tuple = tuple(info_tuple)
                inst_bar1_note.add(info_tuple)

        inst_bar2_note = set()
        for pos in inst_bar2:
            temp_pos_notes = inst_bar2[pos]
            for note in temp_pos_notes:
                info_tuple = [pos]
                if not ignore_pitch:
                    info_tuple.append(note[0])
                if not ignore_duration_for_drum or inst != 128:
                    info_tuple.append(note[1])
                info_tuple = tuple(info_tuple)
                inst_bar2_note.add(info_tuple)

        num_common_notes = len(inst_bar1_note & inst_bar2_note)
        s = num_common_notes / len(inst_bar1_note | inst_bar2_note) if num_common_notes > 0 else 'O'
        o_sim[inst] = s

    return o_sim


def generate_bar_insts_pos_index(bar):
    r = {}
    for idx, item in enumerate(bar):
        notes = item[-1]
        if notes is not None:
            for inst in notes:
                if inst not in r:
                    r[inst] = {}
                r[inst][idx] = notes[inst]
    return r


def construct_bars_info(pos_info, bars_positions):
    bars_note_info = []
    bars_ts_info = []
    num_bars = len(bars_positions)
    for bar_idx in range(num_bars):
        begin, end = bars_positions[bar_idx]
        ts = pos_info[begin][1]
        assert ts is not None
        bars_ts_info.append(ts)

        r = generate_bar_insts_pos_index(pos_info[begin: end])
        bars_note_info.append(r)

    return bars_note_info, bars_ts_info


def cal_for_bar_i_and_j(bar_indices, bars_insts, bars_note_info, bars_ts_info):
    i, j = bar_indices
    bar_i_ts = bars_ts_info[i]
    bar_j_ts = bars_ts_info[j]
    if bar_i_ts != bar_j_ts:
        return None

    r = cal_bar_similarity_basic(
        bars_note_info[i], bars_note_info[j], set(bars_insts[i]), set(bars_insts[j])
    )
    return r


def cal_song_similarity(pos_info, bars_positions, bars_insts, use_multiprocess=True, use_sparse_format=True):
    num_bars = len(bars_positions)
    bars_note_info, bars_ts_info = construct_bars_info(pos_info, bars_positions)

    all_insts = set()
    for bar_insts in bars_insts:
        for inst_id in bar_insts:
            all_insts.add(inst_id)

    inputs = []
    for i in range(num_bars):
        for j in range(i):
            inputs.append((i, j))

    r = {}
    for inst_id in all_insts:
        r[inst_id] = {}
        for i in range(num_bars):
            for j in range(i):
                r[inst_id][(i, j)] = None
    # r: inst_id: {(0, 1): value / None}

    if use_multiprocess:
        with Pool(min(os.cpu_count(), len(inputs))) as pool:
            iterator = iter(
                pool.imap(
                    partial(
                        cal_for_bar_i_and_j,
                        bars_insts=bars_insts,
                        bars_note_info=bars_note_info,
                        bars_ts_info=bars_ts_info,
                    ),
                    inputs
                )
            )
            for i, j in inputs:
                ij_r = next(iterator)
                if ij_r is not None:
                    for inst_id in ij_r:
                        r[inst_id][(i, j)] = ij_r[inst_id]
    else:
        for i, j in inputs:
            ij_r = cal_for_bar_i_and_j(
                (i, j), bars_insts=bars_insts, bars_note_info=bars_note_info, bars_ts_info=bars_ts_info
            )
            if ij_r is not None:
                for inst_id in ij_r:
                    r[inst_id][(i, j)] = ij_r[inst_id]

    if use_sparse_format:
        r = compress_value(r, num_bars)

    return r


def compress_value(data, num_bars):
    for inst_id in data:
        record = data[inst_id]
        new_record = []
        for i in range(num_bars):
            for j in range(i):
                v = record[(i, j)]
                if v == 'O':
                    continue
                new_record.append((i, j, v))
        data[inst_id] = new_record
    return data


def convert_sparse_to_numpy(value, num_bars, ignore_none=True):
    r = {}
    for inst_id in value:
        record = value[inst_id]
        tensor = np.zeros((num_bars, num_bars))
        for i, j, s in record:
            if s is None and ignore_none:
                continue
            tensor[i, j] = s
            tensor[j, i] = s
        for i in range(num_bars):
            tensor[i, i] = 1.0
        r[inst_id] = tensor
    return r


def repr_value(value):
    return convert_dict_key_to_str(value)


def derepr_value(value):
    return convert_dict_key_to_int(value)
