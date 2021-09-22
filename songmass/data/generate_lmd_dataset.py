# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#


import os
import io
import argparse
import copy

import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize


parser = argparse.ArgumentParser(description="LMD data generation")

parser.add_argument('--lmd-data-dir', type=str, default="",
                    help="Lmd data directory")
parser.add_argument('--output-dir', type=str, default="",
                    help="Output directory")


REST_NOTE, SEP, ALIGN = 128.0, '[sep]', '[align]'


duration_vocab = dict([
    (str(x / 100), 129 + i) for i, x in enumerate(list(range(25, 3325, 25)))
])


base_tones = {
    'C' : 0, 'C#': 1, 'D' : 2, 'D#': 3,
    'E' : 4, 'F' : 5, 'F#': 6, 'G' : 7,
    'G#': 8, 'A' : 9, 'A#':10, 'B' :11,
}


all_scales = {
    'major'         : [0, 2, 4, 5, 7, 9, 11],
    'natural_minor' : [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor' : [0, 2, 3, 5, 7, 9, 11],
}


def tones_to_scales(tones):
    """
      Midi to tone name (octave: -5):
        0  1  2  3  4  5  6  7  8  9  10  11
        C  C# D  D# E  F  F# G  G# A  A#  B
        
      Melodic minor scale is ignored. One octave is 12 tones.
    """
    counts = {
        tone: {
            scale: 0 for scale in all_scales
        }
        for tone in base_tones
    }

    if not tones:
        return None
    
    for cur_tone_value in tones:
        for base_tone_name, base_tone_value in base_tones.items():
            for scale_name, scale_notes in all_scales.items():
                if (cur_tone_value - base_tone_value) % 12 in scale_notes:
                    counts[base_tone_name][scale_name] += 1

    frequency = {}
    for base_tone in counts:
        frequency[base_tone] = {}
        for scale_label in counts[base_tone]:
            frequency[base_tone][scale_label] = counts[base_tone][scale_label] / len(tones)
    return frequency


def maximum_likelihood_scale(tones):
    statistic_scales = tones_to_scales(tones)
    a = []
    for base_tone in statistic_scales:
        for scale_label in statistic_scales[base_tone]:
            a.append((base_tone, scale_label, statistic_scales[base_tone][scale_label]))
    a.sort(key=lambda v: v[2], reverse=True)
    return a[0][0], a[0][1], a[0][2]


def normalize_song(tones):
    base_tone, scale_type, scale_score = maximum_likelihood_scale([t for t in tones if t != REST_NOTE])

    if 'major' in scale_type:
        offset = base_tones[base_tone] - base_tones['C']
    else:
        assert 'minor' in scale_type
        offset = base_tones[base_tone] - base_tones['A']

    tones = [t - offset if t != REST_NOTE else REST_NOTE for t in tones]

    most_common_octave = Counter([t // 12 for t in tones]).most_common(1)[0][0]
    offset = 12 * (most_common_octave - 5)

    tones = [t - offset if t != REST_NOTE else REST_NOTE for t in tones]
    return tones


def extract_melody_and_lyric(file_path):
    data = np.load(file_path, allow_pickle=True)[0]
    raw_melody, raw_lyric = data[1], data[2]

    whole_melody, whole_lyric = [], []
    tmp_melody, tmp_lyric = [], []

    # check file is empty or not, and filter empty sentences
    for melody, lyric in zip(raw_melody, raw_lyric):
        if len(melody) > 0 and len(lyric) > 0:
            whole_melody.extend(melody)
            whole_lyric.extend(lyric)

            tmp_melody.append(melody)
            tmp_lyric.append(lyric)
    
    raw_melody = tmp_melody
    raw_lyric = tmp_lyric
    
    if len(whole_melody) == 0 or len(whole_lyric) == 0:
        return [], []

    def generate_sentence(s):
        word_list = []

        for w in s:
            word_list.append(w[0])
            word_list.append(ALIGN)

        word_list.append(SEP)
        return word_list

    lyric = []
    for x in raw_lyric:
        lyric.extend(generate_sentence(x))

    all_notes = []
    for sent in raw_melody:
        for word in sent:
            for syllable in word:
                all_notes.append(syllable[0])
    normalized_notes = normalize_song(all_notes)

    normalized_melody = copy.deepcopy(raw_melody)
    idx = 0
    for i, melody in enumerate(raw_melody):
        for j, word in enumerate(melody):
            for k, syllable in enumerate(word):
                normalized_melody[i][j][k][0] = normalized_notes[idx]
                idx = idx + 1
    
    notes, durations = [], []

    for sent in normalized_melody:
        cur_sent_notes = []
        cur_sent_durations = []
        cur_sent = []
        for word in sent:
            for syllable in word:
                if syllable[2] > 0:
                    pitch = REST_NOTE
                    duration = round(syllable[2] / 0.25) * 0.25
                    cur_sent_notes.append(pitch)
                    cur_sent_durations.append(duration)

                pitch = syllable[0]
                duration = round(syllable[1] / 0.25) * 0.25
                cur_sent_notes.append(pitch)
                cur_sent_durations.append(duration)
            
            cur_sent_notes.append(REST_NOTE) # word-level alignment flag
            cur_sent_durations.append(-2)

        notes.extend(cur_sent_notes)
        durations.extend(cur_sent_durations)

        notes.append(REST_NOTE)
        durations.append(-1) # sentence-level alignment flag

    durations = [min(x, 4.0) for x in durations]
    melody = list(zip(notes, durations))

    return melody, lyric


def check_length(x):
    pitchs = [int(i[0]) for i in x]
    pitchs = list(filter(lambda x : x != REST_NOTE, pitchs))

    cur_num, cur_cnt, max_repeat = -1, -1, 6

    for i in pitchs:
        if i != cur_num:
            if cur_cnt >= max_repeat:
                return False
            cur_num = i
            cur_cnt = 1
        else:
            cur_cnt += 1
    return True


def generate_pitch_duration_sequence(notes):
    ret_list = []
    for x in notes:
        if x[1] == -2:
            ret_list.append(ALIGN)
        elif x[1] == -1:
            ret_list.append(SEP)
        else:
            ret_list.append(str(int(x[0])))
            ret_list.append(str(duration_vocab[str(x[1])]))
    return ret_list


def find_forward(melody_idx, lyric_idx, start, length=250):
    i = start
    for i in range(start + 1, len(melody_idx)):
        if melody_idx[i] - melody_idx[start] > length or lyric_idx[i] - lyric_idx[start] > length:
            i -= 1
            break
    if i == start:
        i += 1
    return i


def find_backward(melody_idx, lyric_idx, start, length=500):
    i = start
    for i in range(start - 1, -1, -1):
        if melody_idx[start] - melody_idx[i] > length or lyric_idx[start] - lyric_idx[i] > length:
            i -= 1
            break
    if i == start:
        i -= 1
    return i


def sliding_window(melody, lyric, max_stride_size=250, max_window_size=500):
    ret_melody, ret_lyric = [], []
    melody_idx = [i for i, x in enumerate(melody) if x == SEP]
    lyric_idx = [i for i, x in enumerate(lyric) if x == SEP]

    assert len(melody_idx) == len(lyric_idx)
    melody_idx.insert(0, -1)
    lyric_idx.insert(0, -1)

    head = 0
    tail = find_forward(melody_idx, lyric_idx, head, max_window_size)
    ret_melody.append(melody[melody_idx[head] + 1 : melody_idx[tail] + 1])
    ret_lyric.append(lyric[lyric_idx[head] + 1 : lyric_idx[tail] + 1])

    while tail != len(melody_idx) - 1:
        tail = find_forward(melody_idx, lyric_idx, tail, max_stride_size)
        head = find_backward(melody_idx, lyric_idx, tail, max_window_size)
        ret_melody.append(melody[melody_idx[head] + 1 : melody_idx[tail] + 1])
        ret_lyric.append(lyric[lyric_idx[head] + 1 : lyric_idx[tail] + 1])
    return ret_melody, ret_lyric


def cut_window(melody, lyric, max_window_size=500):
    ret_melody = []
    ret_lyric = []

    melody_idx = [i for i, x in enumerate(melody) if x == SEP]
    lyric_idx = [i for i, x in enumerate(lyric) if x == SEP]

    assert len(melody_idx) == len(lyric_idx)
    melody_idx.insert(0, -1)
    lyric_idx.insert(0, -1)

    head, tail = 0, 0
    while tail <= len(lyric_idx):
        if tail == len(lyric_idx) and head < tail:
            tail -= 1
            ret_melody.append(melody[melody_idx[head] + 1 : melody_idx[tail] + 1])
            ret_lyric.append(lyric[lyric_idx[head] + 1 : lyric_idx[tail] + 1])
            break
        
        if melody_idx[tail] - melody_idx[head] > max_window_size or lyric_idx[tail] - lyric_idx[head] > max_window_size:
            tail -= 1
            ret_melody.append(melody[melody_idx[head] + 1 : melody_idx[tail] + 1])
            ret_lyric.append(lyric[lyric_idx[head] + 1 : lyric_idx[tail] + 1])
            head = tail

        tail += 1

    return ret_melody, ret_lyric


def lower_fn(x):
    return " ".join(x).lower() + "\n"


def main(args):
    melody_list, lyric_list = [], []
    for file_name in os.listdir(args.lmd_data_dir):
        file_path = os.path.join(args.lmd_data_dir, file_name)
        melody, lyric = extract_melody_and_lyric(file_path)
        
        if len(melody) == 0 or len(lyric) == 0:
            continue

        if not check_length(melody):
            continue

        melody = generate_pitch_duration_sequence(melody)

        melody_list.append(melody)
        lyric_list.append(lyric)

    n = len(melody_list)

    melody_train = melody_list[int(n * 0.2) : ]
    melody_valid = melody_list[int(n * 0.1) : int(n * 0.2)]
    melody_test  = melody_list[int(n * 0.0) : int(n * 0.1)]

    lyric_train = lyric_list[int(n * 0.2) : ]
    lyric_valid = lyric_list[int(n * 0.1) : int(n * 0.2)]
    lyric_test  = lyric_list[int(n * 0.0) : int(n * 0.1)]
    
    # The training data adopts sliding window, 
    # while the valid/test data directly will be performed by clip-off.
    
    sliding_melody_train = []
    sliding_lyric_train = []
    for melody, lyric in zip(melody_train, lyric_train):
        m, l = sliding_window(melody, lyric)
        sliding_melody_train.extend(m)
        sliding_lyric_train.extend(l)

    cut_melody_valid, cut_lyric_valid, song_id_valid = [], [], []
    for i, (melody, lyric) in enumerate(zip(melody_valid, lyric_valid)):
        m, l = cut_window(melody, lyric)
        cut_melody_valid.extend(m)
        cut_lyric_valid.extend(l)
        song_id_valid.extend([i] * len(m))

    cut_melody_test, cut_lyric_test, song_id_test = [], [], []
    for i, (melody, lyric) in enumerate(zip(melody_test, lyric_test)):
        m, l = cut_window(melody, lyric)
        cut_melody_test.extend(m)
        cut_lyric_test.extend(l)
        song_id_test.extend([i] * len(m))

    
    output_dir = args.output_dir
    output_para_dir = os.path.join(output_dir, 'para')

    if not os.path.exists(output_para_dir):
        os.makedirs(output_para_dir)
    
    with open(os.path.join(output_para_dir, 'train.lyric'), 'w') as f:
        lines = list(map(lower_fn, sliding_lyric_train))
        f.writelines(lines)

    with open(os.path.join(output_para_dir, 'train.melody'), 'w') as f:
        lines = list(map(lower_fn, sliding_melody_train))
        f.writelines(lines)

    with open(os.path.join(output_para_dir, 'valid.lyric'), 'w') as f:
        lines = list(map(lower_fn, cut_lyric_valid))
        f.writelines(lines)

    with open(os.path.join(output_para_dir, 'valid.melody'), 'w') as f:
        lines = list(map(lower_fn, cut_melody_valid))
        f.writelines(lines)

    with open(os.path.join(output_para_dir, 'test.lyric'), 'w') as f:
        lines = list(map(lower_fn, cut_lyric_test))
        f.writelines(lines)

    with open(os.path.join(output_para_dir, 'test.melody'), 'w') as f:
        lines = list(map(lower_fn, cut_melody_test))
        f.writelines(lines)

    with open(os.path.join(output_para_dir, 'song_id_valid.txt'), 'w') as f:
        lines = list(map(lambda x : str(x) + '\n', song_id_valid))
        f.writelines(lines)

    with open(os.path.join(output_para_dir, 'song_id_test.txt'), 'w') as f:
        lines = list(map(lambda x : str(x) + '\n', song_id_test))
        f.writelines(lines)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
