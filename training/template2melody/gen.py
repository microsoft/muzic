# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import miditoolkit
import random
import math
import pickle
import hashlib
import numpy as np
from multiprocessing import Pool
from magenta_chord_recognition import infer_chords_for_sequence, _key_chord_distribution,\
    _key_chord_transition_distribution, _CHORDS, _PITCH_CLASS_NAMES, NO_CHORD
import sys
import glob
from tqdm import tqdm


class Item(object):
    def __init__(self, name, start, end, vel=0, pitch=0, track=0, value=''):
        self.name = name
        self.start = start  # start step
        self.end = end  # end step
        self.vel = vel
        self.pitch = pitch
        self.track = track
        self.value = value

    def __repr__(self):
        return f'Item(name={self.name:>10s}, start={self.start:>4d}, end={self.end:>4d}, ' \
               f'vel={self.vel:>3d}, pitch={self.pitch:>3d}, track={self.track:>2d}, ' \
               f'value={self.value:>10s})\n'

    def __eq__(self, other):
        return self.name == other.name and self.start == other.start and \
            self.pitch == other.pitch and self.track == other.track


key_profile = pickle.load(open('key_profile.pickle', 'rb'))
pos_resolution = 4  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 4  # 4 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
ts_filter = False

min_pitch = 48
max_pitch = 72
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


support_chord = list()
for i in range(len(_CHORDS)):
    if i > 0:
        root, kind = _CHORDS[i]
        chord = '%s:%s' % (_PITCH_CLASS_NAMES[root], kind)
    else:
        # NO_CHORD
        chord = _CHORDS[i]
    support_chord.append(chord)

ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))


def normalize_to_c_major(e):
    def get_pitch_class_histogram(notes, use_duration=True, use_velocity=True, normalize=True):
        weights = np.ones(len(notes))
        # Assumes that duration and velocity have equal weight
        if use_duration:
            weights *= [note[4] for note in notes]  # duration
        if use_velocity:
            weights *= [note[5] for note in notes]  # velocity
        histogram, _ = np.histogram([note[3] % 12 for note in notes], bins=np.arange(
            13), weights=weights, density=normalize)
        if normalize:
            histogram /= (histogram.sum() + (histogram.sum() == 0))
        return histogram
    e = [i for i in e if i[2] < 128]
    histogram = get_pitch_class_histogram(e)
    key_candidate = np.dot(key_profile, histogram)
    key_temp = np.where(key_candidate == max(key_candidate))
    major_index = key_temp[0][0]
    minor_index = key_temp[0][1]
    major_count = histogram[major_index]
    minor_count = histogram[minor_index % 12]
    key_number = 0
    if major_count < minor_count:
        key_number = minor_index
        is_major = False
    else:
        key_number = major_index
        is_major = True
    real_key = key_number
    # transposite to C major or A minor
    if real_key <= 11:
        trans = 0 - real_key
    else:
        trans = 21 - real_key
    pitch_shift = trans
    e = [tuple(k if j != 3 else k + pitch_shift for j, k in enumerate(i))
         for i in e]
    return e, is_major


def enc_ts(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def dec_ts(x):
    return ts_list[x]


def enc_dur(x):
    return min(x, duration_max * pos_resolution)


def dec_dur(x):
    return x


def enc_vel(x):
    return x // velocity_quant


def dec_vel(x):
    return (x * velocity_quant) + (velocity_quant // 2)


def enc_tpo(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def dec_tpo(x):
    return 2 ** (x / tempo_quant) * min_tempo


def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator


def gen_dictionary(prefix):
    num = 0
    with open(f'{prefix}/notes.dict.txt', 'w') as f:
        for i in range(bar_max):
            print('Bar_{}'.format(i), num, file=f)
        for i in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            print('Pos_{}'.format(i), num, file=f)
        for i in range(128):
            print('Pitch_{}'.format(i), num, file=f)
        for i in range(1, duration_max * pos_resolution + 1):
            print('Dur_{}'.format(i), num, file=f)

    with open(f'{prefix}/trend.dict.txt', 'w') as f:
        print('MAJ', num, file=f)
        print('MIN', num, file=f)
        for chord in support_chord:
            print(f'Chord_{chord}', num, file=f)
        print('AUT', num, file=f)
        print('HALF', num, file=f)
        print('NOT', num, file=f)
        for beat_idx in range(beat_note_factor):
            print(f'BEAT_{beat_idx}', num, file=f)


def midi_to_encoding(midi_obj):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    key_signature_change_times = len(
        set(i.key_number for i in midi_obj.key_signature_changes))
    assert key_signature_change_times <= 1, 'too many key signature changes: {}'.format(
        key_signature_change_times)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    assert len(notes_start_pos) != 0
    max_pos = min(max(notes_start_pos) + 1, trunc_pos)
    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    allowed_ts = enc_ts(time_signature_reduce(4, 4))
    allowed_ts_list = [enc_ts(time_signature_reduce(i, j))
                       for i, j in [(4, 4)]]
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                cur = enc_ts(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
                assert cur in allowed_ts_list
                pos_to_info[j][1] = allowed_ts
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = enc_tpo(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = enc_ts(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = enc_tpo(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    bar_to_pos = [0]
    for j in range(len(pos_to_info)):
        ts = dec_ts(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
            bar_to_pos.append(bar_to_pos[-1] + measure_length)

    encoding = []
    start_distribution = [0] * pos_resolution
    lead_idx = None
    for idx, inst in enumerate(midi_obj.instruments):
        if inst.name in ['MELODY']:
            lead_idx = idx
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]
            if info[0] >= bar_max or inst.is_drum:
                continue
            encoding.append((info[0], info[2], 128 if inst.is_drum else inst.program,
                             note.pitch + 128 if inst.is_drum else note.pitch,
                             enc_dur(max(1, time_to_pos(note.end - note.start))),
                             enc_vel(note.velocity),
                             info[1], info[3], idx))
    if len(encoding) == 0:
        return list()
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                          math.log2((x / tot)) for x in start_distribution))

    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)

    encoding.sort()
    encoding, is_major = normalize_to_c_major(encoding)

    if is_major:
        target_chords = ['C:', 'C:maj7']
        period_pitch = 0
        period_or_comma_pitchs = [4, 7]
    else:
        target_chords = ['A:m', 'A:m7']
        period_pitch = 9
        period_or_comma_pitchs = [0, 4]

    max_pos = 0
    note_items = []
    for note in encoding:
        max_pos = max(
            max_pos, bar_to_pos[note[0]] + note[1] + dec_dur(note[4]))
        if 0 <= note[3] < 128:
            note_items.append(Item(
                name='On',
                start=bar_to_pos[note[0]] + note[1],
                end=bar_to_pos[note[0]] + note[1] + dec_dur(note[4]),
                vel=dec_vel(note[5]),
                pitch=note[3],
                track=0))
    note_items.sort(key=lambda x: (x.start, -x.end))
    pos_per_chord = pos_resolution * 2
    max_chords = round(max_pos // pos_per_chord + 0.5)
    chords = infer_chords_for_sequence(note_items,
                                       pos_per_chord=pos_per_chord,
                                       max_chords=max_chords,
                                       key_chord_loglik=key_chord_loglik,
                                       key_chord_transition_loglik=key_chord_transition_loglik
                                       )
    boundry = []
    chord_int = 2
    for chord_idx, chord in enumerate(chords[::chord_int]):
        if chord in target_chords:
            cur_pos = chord_idx * pos_per_chord * chord_int
            boundry.append(cur_pos)
    assert len(
        boundry) >= 2, f'segement must start and end in chords: {target_chords}'

    pitch_sum = [0] * 128
    note_cnt = [0] * 128
    for i in encoding:
        if i[2] < 128:
            pitch_sum[i[-1]] += i[3]
            note_cnt[i[-1]] += 1
    avg_pitch = [pitch_sum[i] / note_cnt[i]
                 if note_cnt[i] >= 50 else -1 for i in range(128)]
    if lead_idx is None:
        lead_idx = max(enumerate(avg_pitch), key=lambda x: x[1])[0]
    assert avg_pitch[lead_idx] != -1
    encoding = [i for i in encoding if i[-1] == lead_idx]

    # filter overlap
    allowed_ts = enc_ts(time_signature_reduce(4, 4))
    tmp = []
    for note in encoding:
        if note[6] != allowed_ts:
            continue
        if note[3] < 0 or note[3] > 127:
            continue
        if len(tmp):
            cur_pos = bar_to_pos[note[0]] + note[1]
            last_note = tmp[-1]
            last_pos = bar_to_pos[last_note[0]] + \
                last_note[1] + dec_dur(last_note[4])
            if cur_pos - last_pos >= 0:
                tmp.append(note)
        else:
            tmp.append(note)
    encoding = tmp
    del tmp

    # normalize pitch
    for i in range(len(encoding)):
        if encoding[i][3] < min_pitch:
            encoding[i] = (*encoding[i][:3], min_pitch +
                           encoding[i][3] % 12, *encoding[i][4:])
        elif encoding[i][3] >= max_pitch + 12:
            encoding[i] = (*encoding[i][:3], max_pitch +
                           encoding[i][3] % 12, *encoding[i][4:])

    # infer chords for lead
    lead_notes = []
    for note in encoding:
        max_pos = max(
            max_pos, bar_to_pos[note[0]] + note[1] + dec_dur(note[4]))
        if 0 <= note[3] < 128:
            lead_notes.append(Item(
                name='On',
                start=bar_to_pos[note[0]] + note[1],
                end=bar_to_pos[note[0]] + note[1] + dec_dur(note[4]),
                vel=dec_vel(note[5]),
                pitch=note[3],
                track=0))
    lead_notes.sort(key=lambda x: (x.start, -x.end))
    pos_per_chord = pos_resolution * 2
    max_chords = round(max_pos // pos_per_chord + 0.5)

    lead_chords = infer_chords_for_sequence(lead_notes,
                                            pos_per_chord=pos_per_chord,
                                            max_chords=max_chords,
                                            key_chord_loglik=key_chord_loglik,
                                            key_chord_transition_loglik=key_chord_transition_loglik
                                            )

    boundry_idx = 0
    segments = []
    for enc in encoding:
        if boundry_idx >= len(boundry):
            break
        cur_pos = bar_to_pos[enc[0]] + enc[1]
        while cur_pos >= boundry[boundry_idx]:
            boundry_idx += 1
            if boundry_idx >= len(boundry):
                break
            segments.append([])
        if len(segments):
            segments[-1].append(enc)

    src_str_list = []
    tgt_str_list = []
    max_notes = 200
    min_notes = 5
    last_notes = []
    last_len = 0
    target_len = random.randint(min_notes, max_notes)

    def notes_to_str(raw_notes):
        src_strs, tgt_strs = [], []
        notes_list = []
        cur_pos = None
        for note in raw_notes:
            if len(notes_list) == 0 or bar_to_pos[note[0]] + note[1] - cur_pos > 2 * pos_resolution:
                notes_list.append([])
            notes_list[-1].append(note)
            cur_pos = bar_to_pos[note[0]] + note[1] + dec_dur(note[4])
        for notes in notes_list:
            if len(notes) < min_notes or len(notes) > max_notes:
                continue
            src_words = []
            if is_major:
                src_words.append('MAJ')
            else:
                src_words.append('MIN')
            tgt_words = []
            first_note = notes[0]
            min_bar = first_note[0]
            for note_idx, note in enumerate(notes):

                cur_pos = bar_to_pos[note[0]] + note[1]
                chord_idx = 2 * note[0]
                if note[1] >= 2 * pos_resolution:
                    chord_idx += 1
                cur_chord = lead_chords[chord_idx]
                src_words.append(f'Chord_{cur_chord}')
                if note_idx != len(notes) - 1:
                    nextpos = bar_to_pos[notes[note_idx + 1]
                                         [0]] + notes[note_idx+1][1]
                    if nextpos - cur_pos >= 1.5 * pos_resolution and dec_dur(note[4]) >= pos_resolution:
                        pitch_type = note[3] % 12
                        if nextpos - cur_pos >= 2 * pos_resolution and (pitch_type == period_pitch or
                                                                        (pitch_type in period_or_comma_pitchs and random.random() <= 0.3) or
                                                                        cur_chord in target_chords):
                            src_words.append('AUT')
                        else:
                            src_words.append('HALF')

                    else:
                        src_words.append('NOT')
                else:
                    if dec_dur(note[4]) >= pos_resolution:
                        pitch_type = note[3] % 12
                        if pitch_type == period_pitch or \
                                (pitch_type in period_or_comma_pitchs and random.random() <= 0.3) or\
                                cur_chord in target_chords:
                            src_words.append('AUT')
                        else:
                            src_words.append('HALF')
                    else:
                        src_words.append('NOT')
                beat_idx = note[1] // pos_resolution
                beat_idx = np.clip(beat_idx, 0, beat_note_factor - 1)
                src_words.append(f'BEAT_{beat_idx}')
                tgt_words.append(f'Bar_{note[0] - min_bar}')
                tgt_words.append(f'Pos_{note[1]}')
                tgt_words.append(f'Pitch_{note[3]}')
                tgt_words.append(f'Dur_{note[4]}')
            src_strs.append(' '.join(src_words))
            tgt_strs.append(' '.join(tgt_words))
        return src_strs, tgt_strs

    for segment in segments:
        cur_len = len(segment)
        if cur_len > max_notes or cur_len == 0:
            if max_notes >= last_len >= min_notes:
                src_strs, tgt_strs = notes_to_str(last_notes)
                src_str_list += src_strs
                tgt_str_list += tgt_strs
            last_notes = []
            last_len = 0
            continue

        if (cur_len + last_len) >= target_len:
            if cur_len + last_len <= max_notes:
                src_strs, tgt_strs = notes_to_str(last_notes + segment)
                src_str_list += src_strs
                tgt_str_list += tgt_strs
                last_notes = []
                last_len = 0
                target_len = random.randint(min_notes, max_notes)
            else:
                if max_notes >= last_len >= min_notes:
                    src_strs, tgt_strs = notes_to_str(last_notes)
                    src_str_list += src_strs
                    tgt_str_list += tgt_strs
                last_notes = segment
                last_len = len(segment)
                target_len = random.randint(last_len, max_notes)
        else:
            last_len += cur_len
            last_notes += segment

    if max_notes >= last_len >= min_notes:
        src_strs, tgt_strs = notes_to_str(last_notes)
        src_str_list += src_strs
        tgt_str_list += tgt_strs

    assert len(src_str_list) == len(tgt_str_list)
    return src_str_list, tgt_str_list, get_hash(encoding)


def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode('ascii')).hexdigest()
    return midi_hash


def process(file_name):
    try:
        midi_obj = miditoolkit.midi.parser.MidiFile(file_name)
        # check abnormal values in parse result
        assert all(0 <= j.start < 2 ** 31 and 0 <= j.end < 2 ** 31 for i in midi_obj.instruments for j in
                   i.notes), 'bad note time'
        assert all(0 < j.numerator < 2 ** 31 and 0 < j.denominator < 2 ** 31 for j in
                   midi_obj.time_signature_changes), 'bad time signature value'
        assert 0 < midi_obj.ticks_per_beat < 2 ** 31, 'bad ticks per beat'
    except BaseException as e:
        print('ERROR(PARSE): ' + file_name + ' ' + str(e) + '\n', end='')
        return None
    try:
        src_str_list, tgt_str_list, enc_hash = midi_to_encoding(midi_obj)
        if len(src_str_list) == 0 or len(tgt_str_list) == 0:
            print('ERROR(BLANK): ' + file_name + '\n', end='')
            return None

        print('SUCCESS: ' + file_name + '\n', end='')
        return src_str_list, tgt_str_list, enc_hash
    except BaseException as e:
        # traceback.print_exc()
        print('ERROR(PROCESS): ' + file_name + ' ' + str(e) + '\n', end='')
        return False
    print('ERROR(GENERAL): ' + file_name + '\n', end='')
    return False


def process_with_catch(file_name):
    try:
        return process(file_name)
    except BaseException as e:
        print('ERROR(UNCAUGHT): ' + file_name + '\n', end='')
        return False


def init():
    global key_chord_loglik, key_chord_transition_loglik
    chord_pitch_out_of_key_prob = 0.01
    key_change_prob = 0.001
    chord_change_prob = 0.5
    key_chord_distribution = _key_chord_distribution(
        chord_pitch_out_of_key_prob=chord_pitch_out_of_key_prob)
    key_chord_loglik = np.log(key_chord_distribution)
    key_chord_transition_distribution = _key_chord_transition_distribution(
        key_chord_distribution,
        key_change_prob=key_change_prob,
        chord_change_prob=chord_change_prob)
    key_chord_transition_loglik = np.log(key_chord_transition_distribution)


if __name__ == '__main__':
    assert len(sys.argv) >= 3
    data_path = sys.argv[1:-1]
    name = sys.argv[-1]
    prefix = f'data/{name}'
    os.makedirs(prefix, exist_ok=True)
    file_list = []
    for path in data_path:
        file_list += list(glob.glob(f'{path}/**/*.mid*', recursive=True))
    random.shuffle(file_list)
    gen_dictionary(prefix)
    midi_dict = dict()

    with Pool(24, initializer=init) as p:
        total_file_cnt = len(file_list)
        for sp in ['train', 'valid', 'test']:
            ok_cnt = 0
            all_cnt = 0
            file_list_split = []
            if sp == 'train':  # 98%
                file_list_split = file_list[: 98 * total_file_cnt // 100]
            if sp == 'valid':  # 1%
                file_list_split = file_list[98 * total_file_cnt //
                                            100: 99 * total_file_cnt // 100]
            if sp == 'test':  # 1%
                file_list_split = file_list[99 * total_file_cnt // 100:]
            src_file = f'{prefix}/{sp}.trend'
            tgt_file = f'{prefix}/{sp}.notes'
            result = [p.apply_async(process_with_catch, args=[midi_fn])
                      for midi_fn in file_list_split]

            with open(src_file, 'w') as s, open(tgt_file, 'w') as t:
                for r in tqdm(result):
                    tmp = r.get()
                    if tmp is not None:
                        all_cnt += 1
                        if tmp is not False:
                            src_str_list, tgt_str_list, midi_hash = tmp
                            if deduplicate:
                                duplicated = False
                                dup_file_name = ''
                                if midi_hash in midi_dict:
                                    dup_file_name = midi_dict[midi_hash]
                                    duplicated = True
                                else:
                                    midi_dict[midi_hash] = True
                                if duplicated:
                                    print(f'ERROR(DUPLICATED): {midi_hash}')
                                    continue

                            ok_cnt += 1
                            for output_str in src_str_list:
                                s.write(f'{output_str}\n')
                            for output_str in tgt_str_list:
                                t.write(f'{output_str}\n')
            print('{}: {}/{} ({:.2f}%) midi files successfully processed'.format(sp, ok_cnt, all_cnt,
                                                                                 ok_cnt / all_cnt * 100 if all_cnt
                                                                                 else 0))
