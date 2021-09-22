# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import numpy as np
from dtw import accelerated_dtw
from tqdm import tqdm
import miditoolkit
import sys
import os

min_pitch = 60
max_pitch = 72

ALIGN = '[align]'
SEP = '[sep]'
Duration_vocab = dict([(str(129+i), x/100)
                       for i, x in enumerate(list(range(25, 3325, 25)))])
Duration_vocab_re = dict([(x/100, str(129+i))
                          for i, x in enumerate(list(range(25, 3325, 25)))])
MAX_DUR = int(max(Duration_vocab.keys()))


def seperate(string):
    tmp = [i for i in string.strip().split() if i not in [SEP, ALIGN]]
    return [[tmp[i], tmp[i+1]] for i in range(0, len(tmp), 2)]


def flatten(note_seq, ign_rest=False):
    ret = []
    for note in note_seq:
        pitch = int(note[0])
        duration = Duration_vocab[note[1]]
        if pitch == 128:
            if ign_rest:
                continue
            if len(ret) == 0:
                continue
            ret.extend([ret[-1]] * int(duration*4))

        else:
            if pitch < min_pitch:
                pitch = min_pitch + pitch % 12
            elif pitch >= max_pitch + 12:
                pitch = max_pitch + pitch % 12
            ret.extend([pitch] * int(duration*4))
    return ret


def sample(flat_note_seq, freq=2):
    # 1/16 -> 1
    return [flat_note_seq[i*freq] for i in range(len(flat_note_seq)//freq)]


def grad(flat_note_seq):
    if len(flat_note_seq) == 0:
        return []
    ret = flat_note_seq.copy()
    for i in range(len(ret)-1, 0, -1):
        ret[i] = ret[i] - ret[i-1]
    ret[0] = 0
    return ret[1:]


def get_melody(mid_obj):
    def get_vocab(dur):
        beat = dur // (ticks // 4)
        if beat == 0:
            beat = 1
        beat = beat / 4
        return str(Duration_vocab_re.get(beat, MAX_DUR))

    lyrics = mid_obj.lyrics
    notes = mid_obj.instruments[0].notes
    last_end = notes[0].start
    cur_melody = []
    ticks = mid_obj.ticks_per_beat
    max_lyric = len(lyrics)
    lyric_id = 1
    for note in notes:
        if lyric_id < max_lyric and lyrics[lyric_id].time == note.start:
            cur_melody.append(ALIGN)
            lyric_id += 1
        if note.start != last_end:
            cur_melody.append(str(128))
            cur_melody.append(get_vocab(note.start - last_end))
        cur_melody.append(str(note.pitch))
        cur_melody.append(get_vocab(note.end - note.start))
        last_end = note.end

    cur_melody.append(ALIGN)
    cur_melody.append(SEP)
    new_meldoy = []
    cur_pitch = []
    for i in cur_melody:
        if i == SEP:
            new_meldoy.append(i)
        elif i != ALIGN:
            new_meldoy.append(i)
            if int(i) < 128:
                cur_pitch.append(i)
        elif i == ALIGN:
            if len(cur_pitch):
                new_meldoy.append(i)
            cur_pitch = []
    cur_melody = new_meldoy

    return ' '.join(cur_melody)


if __name__ == '__main__':
    assert len(sys.argv) == 1 + 2
    gt_prefix = sys.argv[1]

    hyp_prefix = sys.argv[2]

    dtw_mean = []
    print()
    print(hyp_prefix)
    for filename in os.listdir(f'{hyp_prefix}/'):
        try:
            hyp_midi = miditoolkit.MidiFile(f'{hyp_prefix}/{filename}')
            hyp = get_melody(hyp_midi)
            gt_midi = miditoolkit.MidiFile(f'{gt_prefix}/{filename}')
            gt = get_melody(gt_midi)
            hyp = seperate(hyp)
            hyp_min = Duration_vocab[str(min([int(i[1]) for i in hyp]))]
            gt = seperate(gt)
            gt_min = Duration_vocab[str(min([int(i[1]) for i in gt]))]

            hyp = flatten(hyp)
            hyp = sample(hyp, freq=int(hyp_min * 4))
            gt = flatten(gt)
            gt = sample(gt, freq=int(gt_min * 4))

            d1 = np.array(gt).reshape(-1, 1)
            d2 = np.array(hyp).reshape(-1, 1)
            d1 = d1 - np.mean(d1)
            d2 = d2 - np.mean(d2)
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(
                d1, d2, dist='euclidean')
            dtw_mean.append(d / len(d2))
        except BaseException as e:
            continue
    print(np.mean(dtw_mean), np.std(dtw_mean))
