# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from tqdm import tqdm
import numpy as np
import traceback
import sys
PITCH_CLASS_NAMES = [
    'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
POS_RESOLUTION = 4
ROOT_pitch = {
    'C': 0,
    'C#': 1,
    'D': 2,
    'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'G': 7,
    'Ab': 8,
    'A': 9,
    'Bb': 10,
    'B': 11
}

_CHORD_KIND_PITCHES = {
    '': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}


def get_tonality(e):
    def get_pitch_class_histogram(notes, use_duration=True, normalize=True):
        weights = np.ones(len(notes))
        if use_duration:
            weights *= [note[3] for note in notes]  # duration
        histogram, _ = np.histogram([note[2] % 12 for note in notes], bins=np.arange(
            13), weights=weights, density=normalize)
        if normalize:
            histogram /= (histogram.sum() + (histogram.sum() == 0))
        return histogram
    e = [i for i in e if i[2] < 128]
    histogram = get_pitch_class_histogram(e)
    major_count = histogram[PITCH_CLASS_NAMES.index('C')]
    minor_count = histogram[PITCH_CLASS_NAMES.index('A')]
    if major_count < minor_count:
        is_major = False
    elif major_count > minor_count:
        is_major = True
    else:
        is_major = None
    return is_major


def fix(items):
    tmp = []
    target_tokens = ['Bar', 'Pos', 'Pitch', 'Dur']
    i = 0
    for item in items:
        if item.split('_')[0] == target_tokens[i]:
            tmp.append(item)
            i = (i + 1) % len(target_tokens)
    return tmp


def get_value(s):
    return s.split('_')[1]


def get_pitch(chord):
    try:
        root, type = chord.split(':')
    except:
        return None
    cur_pitch = []
    for i in _CHORD_KIND_PITCHES[type]:
        cur_pitch.append((ROOT_pitch[root] + i) % 12)
    return cur_pitch


if __name__ == '__main__':
    all_num = 0
    ok_num = 0
    note_num = 0
    beat_num = 0
    chord_num = 0
    struct_num = 0
    struct_num_2 = 0
    struct_num_3 = 0
    pause1_num = 0
    pause2_num = 0
    pause3_num = 0

    tonality_num = 0
    tonality_sum = 0

    chord_sum_2 = 0
    chord_num_2 = 0
    assert len(sys.argv) == 1 + 1
    prefix = sys.argv[1]

    with open(f'{prefix}/test.hyp.txt', 'r') as h, open(f'{prefix}/test.src.txt', 'r') as s:
        for hyp_str, src_str in tqdm(list(zip(h, s))):
            try:
                all_num += 1
                hyp = hyp_str.strip().split()
                hyp = fix(hyp)
                hyp = [[int(get_value(hyp[j])) for j in range(i, i+4)]
                       for i in range(0, len(hyp) // 4 * 4, 4)]

                src = src_str.strip().split()
                is_major = get_tonality(hyp)
                if is_major is not None:
                    tonality_sum += 1
                    if is_major == (src[0] == 'MAJ'):
                        tonality_num += 1
                src = src[1:]
                src = [[get_value(src[i]), src[i+1], int(get_value(src[i+2]))]
                       for i in range(0, len(src), 3)]

                max_pos = 0
                note_items = []

                for idx in range(min(len(hyp), len(src))):
                    hyp_item = hyp[idx]
                    src_item = src[idx]
                    note_num += 1
                    bar, pos, pitch, dur = hyp_item
                    chord, struct, beat = src_item
                    if pos // POS_RESOLUTION == beat:
                        beat_num += 1
                    cur_pitch = get_pitch(chord)
                    if cur_pitch is None or pitch % 12 in cur_pitch:
                        chord_num += 1
                    if idx != len(hyp) - 1:
                        if struct == 'HALF':
                            pause1_num += 1
                        elif struct == 'AUT':
                            pause2_num += 1
                        else:
                            pause3_num += 1
                        next_item = hyp[idx + 1]
                        cur_pos = 4 * POS_RESOLUTION * bar + pos
                        next_pos = 4 * POS_RESOLUTION * \
                            next_item[0] + next_item[1]
                        if next_pos - cur_pos >= POS_RESOLUTION * 1.5 and struct == 'HALF' and dur >= POS_RESOLUTION:
                            struct_num += 1
                        if next_pos - cur_pos >= POS_RESOLUTION * 2.0 and struct == 'AUT' and dur >= POS_RESOLUTION:
                            struct_num_2 += 1
                        if struct == 'NOT':
                            if next_pos - cur_pos < POS_RESOLUTION * 2.0 or dur < POS_RESOLUTION:
                                struct_num_3 += 1
                ok_num += 1
            except:
                continue
    print('TA:', round(tonality_num/tonality_sum, 5))
    print('CA:', round(chord_num/note_num, 5))
    print('RA:', round(beat_num / note_num, 5))
    print('AA:', round((struct_num+struct_num_2+struct_num_3) /
                       (pause1_num + pause2_num+pause3_num), 5))
