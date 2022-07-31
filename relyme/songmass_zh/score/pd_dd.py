# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import numpy as np
import os
import sys
import miditoolkit

ALIGN = '[align]'
SEP = '[sep]'
Duration_vocab = dict([(129+i, x/100)
                       for i, x in enumerate(list(range(25, 3325, 25)))])
Duration_vocab_re = dict([(x/100, 129+i)
                          for i, x in enumerate(list(range(25, 3325, 25)))])
MAX_DUR = int(max(Duration_vocab.keys()))
REST = 128
min_pitch = 60
max_pitch = 72


def get_pitch_count(x):
    cnt = [0] * (128 + 1)
    for pitch in x:
        if pitch != 128:
            if pitch <= min_pitch:
                pitch = min_pitch + pitch % 12
            elif pitch >= max_pitch + 12:
                pitch = max_pitch + pitch % 12
            cnt[pitch] += 1
    return np.array(cnt)

def get_dur_count(x):
    cnt = [0 for i, x in enumerate(list(range(25, 3325, 25)))]
    min_dur = Duration_vocab[min(x)]
    for dur in x:
        cur_dur = Duration_vocab[dur]
        cur_idx = int(cur_dur / min_dur)
        if cur_idx < len(cnt):
            cnt[cur_idx] += 1
        else:
            cnt[-1] += 1
    return np.array(cnt)

def separate(string, use_word=True):
    if not use_word:
        tmp = [i for i in string.strip().split() if i not in [SEP, ALIGN]]
        pitch = []
        dur = []
        for i in range(0, len(tmp), 2):
            pitch.append(int(tmp[i]))
            dur.append(int(tmp[i+1]))
        return pitch, dur
    else:
        tmp = [i for i in string.strip().split() if i not in [SEP]]
        p = []
        d = []
        is_pitch = True
        cur_dur = 0
        for idx in range(len(tmp)):
            if tmp[idx] != ALIGN:
                if is_pitch:
                    p.append(int(tmp[idx]))
                else:
                    cur_dur += Duration_vocab[int(tmp[idx])]
                is_pitch = not is_pitch
            else:
                if cur_dur in Duration_vocab_re:
                    d.append(Duration_vocab_re[cur_dur])
                cur_dur = 0
        if cur_dur > 0 and cur_dur in Duration_vocab_re:
            d.append(Duration_vocab_re[cur_dur])
        return p, d


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


def cal_overlap(gt_d, hyp_d):
    sum_gt = np.sum(gt_d) if np.sum(gt_d) > 0 else 1
    sum_hyp = np.sum(hyp_d) if np.sum(hyp_d) > 0 else 1
    gt_d = gt_d.astype(np.float32) / sum_gt
    hyp_d = hyp_d.astype(np.float32) / sum_hyp
    diff = np.abs(gt_d - hyp_d)
    overlap = (gt_d + hyp_d - diff) / 2
    return np.sum(overlap)

def get_pd_dd(gt, hyp):
    pitch_overlap = []
    dur_overlap = []
    # print(f'hyp: {hyp}   gt: {gt}')
    
    try:
        hyp_midi = miditoolkit.MidiFile(hyp)
        hyp = get_melody(hyp_midi)
        gt_midi = miditoolkit.MidiFile(gt)
        gt = get_melody(gt_midi)
        hyp_pitch, hyp_dur = separate(hyp)
        gt_pitch, gt_dur = separate(gt)
        pitch_overlap.append(cal_overlap(
            get_pitch_count(gt_pitch), get_pitch_count(hyp_pitch)))
        dur_overlap.append(cal_overlap(
            get_dur_count(gt_dur), get_dur_count(hyp_dur)))
    except Exception:
        print("FAILED")
    
    pitch_overlap = np.array(pitch_overlap)
    dur_overlap = np.array(dur_overlap)
    
    return pitch_overlap, dur_overlap

if __name__ == '__main__':
    assert len(sys.argv) == 1 + 2
    gt_prefix = sys.argv[1]
    hyp_prefix = sys.argv[2]

    pitch_overlap = []
    dur_overlap = []
    print(f'hyp: {hyp_prefix}   gt: {gt_prefix}')
    print(os.listdir(f'{hyp_prefix}/'))
    for filename in os.listdir(f'{hyp_prefix}/'):
        try:
            hyp_midi = miditoolkit.MidiFile(f'{hyp_prefix}/{filename}')
            hyp = get_melody(hyp_midi)
            gt_midi = miditoolkit.MidiFile(f'{gt_prefix}/{filename}')
            gt = get_melody(gt_midi)
            hyp_pitch, hyp_dur = separate(hyp)
            gt_pitch, gt_dur = separate(gt)
            pitch_overlap.append(cal_overlap(
                get_pitch_count(gt_pitch), get_pitch_count(hyp_pitch)))
            dur_overlap.append(cal_overlap(
                get_dur_count(gt_dur), get_dur_count(hyp_dur)))
        except BaseException as e:
            continue
    pitch_overlap = np.array(pitch_overlap)
    dur_overlap = np.array(dur_overlap)
    print('PITCH', np.mean(pitch_overlap), np.std(pitch_overlap))
    print('DUR', np.mean(dur_overlap), np.std(dur_overlap))
    print()
