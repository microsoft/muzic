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
from utils import *
import json, shutil

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


# [left, right)
average_pitches = [[0, 57], [57, 65], [65, 72], [72, 128]] # [A3, A3#-E4, F4-B4, C5]
spans = [[0, 8], [8, 15], [15, 128]]


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
    
    with open(f'{prefix}/melody.dict.txt', 'w') as f:
        for i in range(bar_max):
            print('Bar_{}'.format(i), num, file=f)
        for i in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            print('Pos_{}'.format(i), num, file=f)
        for i in range(128):
            print('Pitch_{}'.format(i), num, file=f)
        for i in range(1, duration_max * pos_resolution + 1):
            print('Dur_{}'.format(i), num, file=f)
        for chord in support_chord:
            print(f'Chord_{chord}', num, file=f)
        print('AUT', num, file=f)
        print('HALF', num, file=f)
        print('NOT', num, file=f)
        for beat_idx in range(beat_note_factor):
            print(f'BEAT_{beat_idx}', num, file=f)
        
        print('[sep]', num, file=f)
        print('[div]', num, file=f)


def midi_to_encoding(midi_obj, phrase_range, period_range, infer_chord=False):
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
    is_major = True

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
            last_st_pos = bar_to_pos[last_note[0]] + last_note[1]
            last_et_pos = bar_to_pos[last_note[0]] + last_note[1] + last_note[4]
            if cur_pos - last_st_pos >= last_note[4] * (2 / 3):
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

    if infer_chord:
        lead_chords = infer_chords_for_sequence(lead_notes,
                                                pos_per_chord=pos_per_chord,
                                                max_chords=max_chords,
                                                key_chord_loglik=key_chord_loglik,
                                                key_chord_transition_loglik=key_chord_transition_loglik
                                                )
    else:
        def assign_chords_for_sequence_by_template(phrase_range, pos_per_chord, max_chords):
            chords_mapping = {1: 'C:', 2: 'D:m', 3:'E:m', 4: 'F:', 5: 'G:', 6: 'A:m', 7: 'B:dim'}
            # Maximum length of chord sequence to infer.
            _MAX_NUM_CHORDS = 1000
            lead_chords = []
            beats = [pos_per_chord * i for i in range(max_chords)]
            if len(beats) == 0:
                raise Exception('max chords should > 0')
            num_chords = len(beats)
            if num_chords > _MAX_NUM_CHORDS:
                raise Exception(
                    'NoteSequence too long for chord inference: %d frames' % num_chords)

            for i, item in enumerate(phrase_range):
                st, et = item[0], item[1]
                phrase = item[2]
                chords = item[3]
                chords = [x for sublist in chords for x in sublist]
                assert(len(chords) == et - st)
                for j, chord in enumerate(chords):
                    lead_chords.extend([chords_mapping[chord], chords_mapping[chord]])

            return lead_chords
        lead_chords = assign_chords_for_sequence_by_template(phrase_range, pos_per_chord, max_chords)

    segments = segmentation(encoding, phrase_range)
    src_str_list = []
    tgt_str_list = []
    max_notes = 200
    min_notes = 5
    last_notes = []
    last_len = 0
    target_len = random.randint(min_notes, max_notes)

    def notes_to_str(raw_notes, phrase_range, period_range):
        src_strs, tgt_strs = [], []
        notes_list = []
        cur_pos = None
        for note in raw_notes:
            if len(notes_list) == 0:
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

                src_words.append(f'Bar_{note[0] - min_bar}')
                src_words.append(f'Pos_{note[1]}')

                cur_pos = bar_to_pos[note[0]] + note[1]
                chord_idx = 2 * note[0]
                if note[1] >= 2 * pos_resolution:
                    chord_idx += 1
                cur_chord = lead_chords[chord_idx]
                src_words.append(f'Chord_{cur_chord}')
                # assign cadence
                cur_bar = note[0]
                cadence = None
                if note_idx == len(notes) - 1:
                    for x in period_range:
                        if note[0] == x[1] - 1:
                            cadence = 'AUT'
                    if cadence is None:
                        cadence = 'HALF'
                else:
                    nextpos = bar_to_pos[notes[note_idx + 1]
                                         [0]] + notes[note_idx+1][1]
                    if nextpos - cur_pos >= 1.5 * pos_resolution and dec_dur(note[4]) >= pos_resolution:
                        pitch_type = note[3] % 12
                        if nextpos - cur_pos >= 2 * pos_resolution and (pitch_type == period_pitch or
                                                                        (pitch_type in period_or_comma_pitchs and random.random() <= 0.3) or
                                                                        cur_chord in target_chords):
                            cadence = 'AUT'
                        else:
                            cadence = 'HALF'

                    else:
                        cadence = 'NOT'
                
                src_words.append(cadence)
                beat_idx = note[1] // pos_resolution
                beat_idx = np.clip(beat_idx, 0, beat_note_factor - 1)
                src_words.append(f'Dur_{note[4]}')
                # src_words.append(f'BEAT_{beat_idx}')
                tgt_words.append(f'Bar_{note[0] - min_bar}')
                tgt_words.append(f'Pos_{note[1]}')
                tgt_words.append(f'Pitch_{note[3]}')
                tgt_words.append(f'Dur_{note[4]}')
            src_strs.append(' '.join(src_words))
            tgt_strs.append(' '.join(tgt_words))
        return src_strs, tgt_strs

    # add template features to differentiate A/B	
    def get_avg_pitch(seg):	
        s = sum([x[3] for x in seg])	
        return s / len(seg)	
    	
    def get_span(seg):	
        low_pitch, high_pitch = 128, 0	
        for x in seg:	
            low_pitch = min(low_pitch, x[3])	
            high_pitch = max(high_pitch, x[3])	
        return high_pitch, low_pitch	
    	
    def get_trend(seg):	
        diff = seg[-1][3] - seg[0][3]	
        if diff > 0:	
            return 'UP'	
        if diff == 0:	
            return 'STILL'	
        if diff < 0:	
            return 'DOWN'

    for segment in segments:
        cur_len = len(segment)
        if cur_len < max_notes and cur_len > min_notes:

            avg_pitch = get_avg_pitch(segment)	
            high_pitch, low_pitch = get_span(segment)	
            span = high_pitch - low_pitch	
            src_strs, tgt_strs = notes_to_str(segment, phrase_range, period_range)

            assert(len(src_strs) == 1)	
            assert(len(tgt_strs) == 1)	
            src_strs = src_strs[0]
            tgt_strs = tgt_strs[0]

            for pair in average_pitches:
                if avg_pitch >= pair[0] and avg_pitch < pair[1]:
                    src_strs = 'AVGPITCH_{}_{} '.format(pair[0], pair[1]) + src_strs

            for pair in spans:
                if span >= pair[0] and span < pair[1]:
                    src_strs = 'SPAN_{}_{} '.format(pair[0], pair[1]) + src_strs

            src_str_list += [src_strs]	
            tgt_str_list += [tgt_strs]
            
    assert len(src_str_list) == len(tgt_str_list)
    src_str, tgt_str, res_str = combine_with_sep(src_str_list, tgt_str_list)
    return src_str, tgt_str, res_str, get_hash(encoding)

def load_template(fn):
    f = open(fn)
    template = json.load(f)
    phrase_struct = template['phrase structure']
    dev_struct = template['dev structure']
    chord_progress = template['chord progress']
    phrase_range = []
    period_range = []
    st = 0
    for l in phrase_struct:
        period_range.append([st, -1, l[0][:-1].upper()])
        for phrase in l:
            methods = dev_struct[phrase][0]
            length = int(dev_struct[phrase][1])
            phrase_length = len(methods.split('|')) * length
            et = st + phrase_length
            chord = chord_progress[phrase]

            phrase_range.append([st, et, phrase, chord])           
            st = et
        period_range[-1][1] = et

    f.close()
    return phrase_range, period_range
    
def segmentation(encoding, phrase_range):
    segments = [[] for i in range(len(phrase_range))]
    i, j = 0, 0

    while i < len(encoding) and j < len(phrase_range):
        cur_bar = encoding[i][0]
        st, et = phrase_range[j][0], phrase_range[j][1]
        phrase = phrase_range[j][2]
        if cur_bar >= st and cur_bar < et:
            segments[j].append(encoding[i])
            i += 1
        else:
            j += 1
        
    return segments
        


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
        phrase_range, period_range = load_template('\\'.join(file_name.split('\\')[:-1] + ['template.json']))
        src_str_list, tgt_str_list, res_str_list, enc_hash = midi_to_encoding(midi_obj, phrase_range, period_range)
        sep_count = len([x for x in res_str_list.split(' ') if x == SEP])
        if len(src_str_list) == 0 or len(tgt_str_list) == 0 or sep_count != len(phrase_range):
            print('ERROR(BLANK): ' + file_name + '\n', end='')
            return None

        print('SUCCESS: ' + file_name + '\n', end='')
        return src_str_list, tgt_str_list, res_str_list, enc_hash, phrase_range
    except BaseException as e:
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

def combine_with_sep(src_str_list, tgt_str_list):
    src_str, tgt_str = '', ''
    res_str = ''
    for src_item, tgt_item in zip(src_str_list, tgt_str_list):
        src_str += src_item + ' [sep] '
        tgt_str += tgt_item + ' [sep] '
        res_str += src_item + ' [div] ' + tgt_item + ' [sep] '
    
    src_str = src_str[:-1]
    tgt_str = tgt_str[:-1]
    res_str = res_str[:-1]
    return src_str, tgt_str, res_str

SEP = '[sep]'
DIV = '[div]'

# each sample is a sentence
def extract_notes(sample):
    sample = sample.split()
    sep_idxes = [i for i,x in enumerate(sample) if x==SEP]
    div_idxes = [i for i,x in enumerate(sample) if x==DIV]

    sentences = []
    for sidx, didx in zip(sep_idxes, div_idxes):
        sentences.append(['<s>'] + sample[didx + 1:sidx] + ['</s>'])
    
    return sentences

if __name__ == '__main__':

    root_path = sys.argv[1]
    song_id = sys.argv[2]
    prefix = sys.argv[3]
    prefix = os.path.join(prefix, song_id)

    data_path = os.path.join(root_path, song_id)
    template_prefix = os.path.join(prefix, 'template')
    os.makedirs(prefix, exist_ok=True)
    os.makedirs(template_prefix, exist_ok=True)

    midi_dict = dict()
    init()
    sp = 'test'
    midi_fn = os.path.join(data_path, f'{song_id}-melody.mid')

    res = process_with_catch(midi_fn)
    if res:
        src_str, tgt_str, res_str, midi_hash, phrase_range = res

        basename = midi_fn.split('\\')[-1][:-4]
        res_file = f'{prefix}\\{sp}-{basename}'
        song_id = basename.split('-')[0]
        shutil.copy2(os.path.join(data_path, 'template.json'), os.path.join(template_prefix, 'template.json'))
        
        # mask phrase
        phrase_set = list(set([x[2] for x in phrase_range]))
        visited = []
        src_sep_positions = [i for i, x in enumerate(src_str.split(' ')) if x == SEP]
        src_sep_positions.insert(0, -1)
        tgt_sep_positions = [i for i, x in enumerate(tgt_str.split(' ')) if x == SEP]
        tgt_sep_positions.insert(0, -1)
        parallels = []
        src_str = src_str.split(' ')
        tgt_str = tgt_str.split(' ')
        phrase_set.sort()
        for tgt_phrase in phrase_set:
            replace_phrase_ids = []
            for i, (st, et, phrase, chords) in enumerate(phrase_range):
                if phrase == tgt_phrase:
                    replace_phrase_ids.append(i)
            
            new_src_str = []
            new_tgt_str = []
            for i in range(len(tgt_sep_positions) - 1):
                mask_start, mask_end = tgt_sep_positions[i] + 1, tgt_sep_positions[i + 1] + 1
                temp_start, temp_end = src_sep_positions[i] + 1, src_sep_positions[i + 1] + 1
                if i in replace_phrase_ids:
                    src_str_tmp = src_str[temp_start:temp_end]         
                    new_src_str += src_str_tmp
                    new_tgt_str += tgt_str[mask_start:mask_end]
                else:
                    new_src_str += tgt_str[mask_start:mask_end]

            new_src_str = ' '.join(new_src_str)
            new_tgt_str = ' '.join(new_tgt_str)

            parallels.append(((tgt_phrase, st, et, 0), new_src_str, new_tgt_str))

            with open(res_file + '-' + tgt_phrase + '-update.template', 'w') as f1:
                f1.write(f'{new_src_str}\n')
            with open(res_file + '-' + tgt_phrase + '-update.melody', 'w') as f1:
                f1.write(f'{new_tgt_str}\n')



