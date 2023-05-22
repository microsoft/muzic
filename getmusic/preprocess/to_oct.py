# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import sys
import io
import zipfile
import miditoolkit
import time
import math
import signal
import hashlib
from multiprocessing import Pool, Lock, Manager
import random
import numpy as np
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from getmusic.utils.midi_config import *
import pickle
from getmusic.utils.magenta_chord_recognition import infer_chords_for_sequence, _key_chord_distribution,\
    _key_chord_transition_distribution, _CHORDS, _PITCH_CLASS_NAMES, NO_CHORD

pos_in_bar = beat_note_factor * max_notes_per_bar * pos_resolution
data_zip = None
output_file = None

key_profile = pickle.load(open('getmusic/utils/key_profile.pickle', 'rb'))

lock_file = Lock()
lock_write = Lock()
lock_set = Lock()
manager = Manager()
midi_dict = manager.dict()


# (0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo)
# (Measure, TimeSig)
# (Pos, Tempo)
# Percussion: Program=128 Pitch=[128,255]
root_dict = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
kind_dict = {'null': 0, 'm': 1, '+': 2, 'dim': 3, 'seven': 4, 'maj7': 5, 'm7': 6, 'm7b5': 7}

ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)

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


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)

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

def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def e2t(x):
    return ts_list[x]


def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]


def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]


def v2e(x):
    return x // velocity_quant


def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)


def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def e2b(x):
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

def writer(file_name, output_str_list):
    # note: parameter "file_name" is reserved for patching
    with open(output_file, 'a') as f:
        for output_str in output_str_list:
            if '<2-80>' in output_str and '<2-129>' in output_str and (('<2-0>' in output_str) or ('<2-25>' in output_str) or ('<2-32>' in output_str) or ('<2-48>' in output_str) or ('<2-128>' in output_str)):
                f.write(output_str + '\n')

def gen_dictionary(file_name):
    num = 0
    with open(file_name, 'w') as f:
        for j in range(bar_max):
            print('<0-{}>'.format(j), num, file=f)
        for j in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            print('<1-{}>'.format(j), num, file=f)
        for j in range(max_inst + 1 + 1):
            # max_inst + 1 for percussion
            print('<2-{}>'.format(j), num, file=f)
        for j in range(2 * max_pitch + 1 + 1):
            # max_pitch + 1 ~ 2 * max_pitch + 1 for percussion
            print('<3-{}>'.format(j), num, file=f)
        for j in range(duration_max * pos_resolution):
            print('<4-{}>'.format(j), num, file=f)
        for j in range(v2e(max_velocity) + 1):
            print('<5-{}>'.format(j), num, file=f)
        for j in range(len(ts_list)):
            print('<6-{}>'.format(j), num, file=f)
        for j in range(b2e(max_tempo) + 1):
            print('<7-{}>'.format(j), num, file=f)

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

    histogram = get_pitch_class_histogram([i for i in e if i[2] < 128])
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

    # _e = []
    # for i in e:
    #     for j, k in enumerate(i):
    #         if i[2] == 128:
    #             _e.append(i)
    #         else:

    e = [tuple(k + pitch_shift if j == 3 and i[2] != 128 else k for j, k in enumerate(i))
         for i in e]
    return e, is_major

def MIDI_to_encoding(midi_obj):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = min(max(notes_start_pos) + 1, trunc_pos)
    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes # [TimeSignature(numerator=4, denominator=4, time=0)]
    tpc = midi_obj.tempo_changes # [TempoChange(tempo=120.0, time=0)]
    
    # filter tempo and ts change
    if len(tsc) > 1 or len(tpc) > 1:
        return ['welcome use my code']

    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
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
    encoding = []
    start_distribution = [0] * pos_resolution

    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]
            duration = d2e(time_to_pos(note.end) - time_to_pos(note.start))
            encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                             1 if inst.is_drum else note.pitch, duration, v2e(note.velocity), info[1], info[3]))
    if len(encoding) == 0:
        return list()
        
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                          math.log2((x / tot)) for x in start_distribution))
    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)
    
    # normalize
    encoding.sort()
    encoding, is_major = normalize_to_c_major(encoding)

    # extract chords
    max_pos = 0
    note_items = []
    for note in encoding:
        if 0 <= note[3] < 128:
            ts = e2t(note[6])
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
            max_pos = max(
                max_pos, measure_length * note[0] + note[1] + e2d(note[4]))
            note_items.append(Item(
                name='On',
                start = measure_length * note[0] + note[1],
                end = measure_length * note[0] + note[1] + e2d(note[4]),
                vel=e2v(note[5]),
                pitch=note[3],
                track=0))
    note_items.sort(key=lambda x: (x.start, -x.end))
    pos_per_chord = measure_length
    max_chords = round(max_pos // pos_per_chord + 0.5)
    chords = infer_chords_for_sequence(note_items,
                                       pos_per_chord=pos_per_chord,
                                       max_chords=max_chords,
                                       key_chord_loglik=key_chord_loglik,
                                       key_chord_transition_loglik=key_chord_transition_loglik
                                       )
    
    bar_idx = 0
    for c in chords:
        r, k = c.split(':')
        if k == '':
            k = 'null'
        elif k == '7':
            k = 'seven'
        encoding.append((bar_idx, 0, 129, root_dict[r], kind_dict[k], 0, t2e(time_signature_reduce(4, 4)), 0))
        bar_idx += 1

    encoding.sort()
    return encoding

def encoding_to_MIDI(encoding):
    # TODO: filter out non-valid notes and error handling
    bar_to_timesig = [list()
                      for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = e2t(bar_to_timesig[i])
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = (i[3] - 128 if program == 128 else i[3])
        duration = get_tick(0, e2d(i[4]))
        if duration == 0:
            duration = 1
        end = start + duration
        velocity = e2v(i[5])
        midi_obj.instruments[program].notes.append(miditoolkit.containers.Note(
            start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj

def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode('ascii')).hexdigest()
    return midi_hash

def F(file_name):
    try:
        with timeout(seconds=600):
            midi_obj = miditoolkit.midi.parser.MidiFile(file_name)
        # check abnormal values in parse result
        assert all(0 <= j.start < 2 ** 31 and 0 <= j.end < 2 **
                   31 for i in midi_obj.instruments for j in i.notes), 'bad note time'
        assert all(0 < j.numerator < 2 ** 31 and 0 < j.denominator < 2 **
                   31 for j in midi_obj.time_signature_changes), 'bad time signature value'
        assert 0 < midi_obj.ticks_per_beat < 2 ** 31, 'bad ticks per beat'
    except BaseException as e:
        print('ERROR(PARSE): ' + file_name + ' ' + str(e) + '\n', end='')
        return None
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    
    no_empty_tracks = {'80':0,'32':0,'128':0,'25':0,'0':0,'48':0}
    for inst in midi_obj.instruments:
        no_empty_tracks[str(inst.program)] = 1

    if no_empty_tracks['80'] == 0 or sum(no_empty_tracks.values()) <= 1:
        print('ERROR(BAD TRACKS): ' + file_name + '\n', end='')
        return False
    try:
        e = MIDI_to_encoding(midi_obj)
        
        if len(e) == 0:
            print('ERROR(BLANK): ' + file_name + '\n', end='')
            return None

        if len(e) == 1:
            print('ERROR(TEMPO CHANGE): ' + file_name + '\n', end='')
            return False

        # if ts_filter:
        allowed_ts = t2e(time_signature_reduce(4, 4))
        if not all(i[6] == allowed_ts for i in e):
            print('ERROR(TSFILT): ' + file_name + '\n', end='')
            return None
        if deduplicate:
            duplicated = False
            dup_file_name = ''
            midi_hash = '0' * 32
            try:
                midi_hash = get_hash(e)
            except BaseException as e:
                pass
            lock_set.acquire()
            if midi_hash in midi_dict:
                dup_file_name = midi_dict[midi_hash]
                duplicated = True
            else:
                midi_dict[midi_hash] = file_name
            lock_set.release()
            if duplicated:
                print('ERROR(DUPLICATED): ' + midi_hash + ' ' +
                      file_name + ' == ' + dup_file_name + '\n', end='')
                return None
        output_str_list = []
        sample_step = max(round(sample_len_max / sample_overlap_rate), 1)
        for p in range(0, len(e), sample_step):
            L = p
            R = min(p + sample_len_max, len(e)) - 1
            bar_index_list = [e[i][0]
                              for i in range(L, R + 1) if e[i][0] is not None]
            bar_index_min = 0
            bar_index_max = 0
            if len(bar_index_list) > 0:
                bar_index_min = min(bar_index_list)
                bar_index_max = max(bar_index_list)
                
            # to make bar index start from 0
            bar_index_offset = -bar_index_min
            e_segment = []
            for i in e[L: R + 1]:
                if i[0] is None or i[0] + bar_index_offset < bar_max:
                    e_segment.append(i)
                else:
                    break
            tokens_per_note = 8
            output_words = ([('<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) if k is not None else '<unk>') for i in e_segment for j, k in enumerate(i)])  # tokens_per_note - 1 for append_eos functionality of binarizer in fairseq
            output_str_list.append(' '.join(output_words))

        # no empty
        if not all(len(i.split()) > tokens_per_note * 2 - 1 for i in output_str_list):
            print('ERROR(ENCODE): ' + file_name + '\n', end='')
            return False
        try:
            lock_write.acquire()
            writer(file_name, output_str_list)
        except BaseException as e:
            print('ERROR(WRITE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        finally:
            lock_write.release()
        print('SUCCESS: ' + file_name + '\n', end='')
        return True
    except BaseException as e:
        print('ERROR(PROCESS): ' + file_name + ' ' + str(e) + '\n', end='')
        return False
    print('ERROR(GENERAL): ' + file_name + '\n', end='')
    return False

def G(file_name):
    try:
        return F(file_name)
    except BaseException as e:
        print('ERROR(UNCAUGHT): ' + file_name + '\n', end='')
        return False

def str_to_encoding(s):
    encoding = [int(i[3: -1]) for i in s.split() if 's' not in i]
    tokens_per_note = 8
    assert len(encoding) % tokens_per_note == 0
    encoding = [tuple(encoding[i + j] for j in range(tokens_per_note))
                for i in range(0, len(encoding), tokens_per_note)]
    return encoding

def encoding_to_str(e):
    bar_index_offset = 0
    p = 0
    tokens_per_note = 8
    return ' '.join((['<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) for i in e[p: p +\
                sample_len_max] if i[0] + bar_index_offset < bar_max for j, k in enumerate(i)]))   # 8 - 1 for append_eos functionality of binarizer in fairseq

if __name__ == '__main__':
    data_path = sys.argv[1]
    prefix = sys.argv[2]
    if os.path.exists(prefix):
        print('Output path {} already exists! Please delete existing files'.format(prefix))
        sys.exit(0)
    os.system('mkdir -p {}'.format(prefix))
    
    
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        for filename in filenames:
            if filename.endswith('.mid'): 
                file_list.append(os.path.join(dirpath, filename))
    file_list.sort()
    # random.shuffle(file_list)

    # gen_dictionary('{}/dict.txt'.format(prefix))
    ok_cnt = 0
    all_cnt = 0

    total_file_cnt = len(file_list)
    file_list_split = file_list
    output_file = '{}/oct.txt'.format(prefix)
    with Pool(pool_num) as p:
        result = list(p.imap_unordered(G, file_list_split))
        all_cnt += sum((1 if i is not None else 0 for i in result))
        ok_cnt += sum((1 if i is True else 0 for i in result))
    output_file = None
    print('{}/{} ({:.2f}%) MIDI files successfully processed'.format(ok_cnt,
                                                                     all_cnt, ok_cnt / all_cnt * 100))
    
    