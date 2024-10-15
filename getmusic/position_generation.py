import argparse
import os
import warnings
import time
import torch
from getmusic.modeling.build import build_model
from getmusic.data.build import build_dataloader
from getmusic.utils.misc import seed_everything, merge_opts_to_config, modify_config_for_debug
from getmusic.utils.io import load_yaml_config
from getmusic.engine.logger import Logger
from getmusic.engine.solver import Solver
from getmusic.distributed.launch import launch
import datetime
import numpy  as np
import pickle
import miditoolkit
import math
from getmusic.utils.midi_config import *
from getmusic.utils.magenta_chord_recognition import infer_chords_for_sequence, _key_chord_distribution,\
    _key_chord_transition_distribution, _CHORDS, _PITCH_CLASS_NAMES, NO_CHORD

NODE_RANK = os.environ['INDEX'] if 'INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = (os.environ['CHIEF_IP'], 22275) if 'CHIEF_IP' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
NUM_NODE = os.environ['HOST_NUM'] if 'HOST_NUM' in os.environ else 1

inst_to_row = { '80':0, '32':1, '128':2,  '25':3, '0':4, '48':5, '129':6}
prog_to_abrv = {'0':'P','25':'G','32':'B','48':'S','80':'M','128':'D'}
track_name = {'M':'lead', 'B':'bass','D':'drum', 'G':'guitar', 'P':'piano', 'S':'string'}

root_dict = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
kind_dict = {'null': 0, 'm': 1, '+': 2, 'dim': 3, 'seven': 4, 'maj7': 5, 'm7': 6, 'm7b5': 7}
root_list = list(root_dict.keys())
kind_list = list(kind_dict.keys())

_CHORD_KIND_PITCHES = {
    'null': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    'seven': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}

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

tokens_to_ids = {}
ids_to_tokens = []
pad_index = None
empty_index = None


key_profile = pickle.load(open('getmusic/utils/key_profile.pickle', 'rb'))

pos_in_bar = beat_note_factor * max_notes_per_bar * pos_resolution


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

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training script')
    parser.add_argument('--config_file', type=str, default='configs/train.yaml', 
                        help='path of config file')
    parser.add_argument('--name', type=str, default='inference_cache', 
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file') 
    parser.add_argument('--output', type=str, default='cache', 
                        help='directory to save the results')    
    parser.add_argument('--log_frequency', type=int, default=10, 
                        help='print frequency (default: 10)')
    parser.add_argument('--load_path', type=str, default=None,
                        help='path to model that need to be loaded, '
                             'used for loading pretrained model') 
    parser.add_argument('--resume_name', type=str, default=None,
                        help='resume one experiment with the given name')
    parser.add_argument('--auto_resume', action='store_true',
                        help='automatically resume the training')

    # args for ddp
    parser.add_argument('--num_node', type=int, default=NUM_NODE,
                        help='number of nodes for distributed training') 
    parser.add_argument('--ngpus_per_node', type=int, default=8,
                        help='number of gpu on one node')
    parser.add_argument('--node_rank', type=int, default=NODE_RANK,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default=DIST_URL, 
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--sync_bn', action='store_true', 
                        help='use sync BN layer')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')
    parser.add_argument('--timestamp', action='store_true', # default=True,
                        help='use tensorboard for logging')
    # args for random
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', action='store_true', 
                        help='set cudnn.deterministic True')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='automatic mixture of precesion')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='set as debug mode')
    
    parser.add_argument('--do_sample', action='store_false', default=True)
    parser.add_argument('--file_path', type=str, default=None)
    parser.add_argument('--skip_step', type=int, default=0)
    parser.add_argument('--decode_chord', action='store_true', default=False)
    parser.add_argument('--no_ema', action='store_false', default=True)
    
    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )  

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.resume_name is not None:
        args.name = args.resume_name
        args.config_file = os.path.join(args.output, args.resume_name, 'configs', 'config.yaml')
        args.auto_resume = True
    else:
        if args.name == '':
            args.name = os.path.basename(args.config_file).replace('.yaml', '')
        if args.timestamp:
            assert not args.auto_resume, "for timstamp, auto resume is hard to find the save directory"
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            args.name = time_str + '-' + args.name

    if args.debug:
        args.name = 'debug'
        if args.gpu is None:
            args.gpu = 0
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')
    args.save_dir = os.path.join(args.output, args.name, now)

    return args

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

    e = [tuple(k + pitch_shift if j == 3 and i[2] != 128 else k for j, k in enumerate(i))
         for i in e]
    return e, is_major, pitch_shift

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
    return math.floor(2 ** (x / tempo_quant) * min_tempo)

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

def MIDI_to_encoding(midi_obj, with_chord=True):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = max(notes_start_pos) + 1

    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes # [TimeSignature(numerator=4, denominator=4, time=0)]
    tpc = midi_obj.tempo_changes # [TempoChange(tempo=120.0, time=0)]
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
    for j in range(len(pos_to_info)): # 它这里是不管这个位置有没有音符，都占个位
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1] # 比如一个3/4的ts，一个4/4的小节有16pos，所以3/4一小节就有12
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:  # 如果cnt>了measure长度，就是下一个小节了，cnt清零，bar index加一
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
            
    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue

            info = pos_to_info[time_to_pos(note.start)]
            duration = d2e(time_to_pos(note.end) - time_to_pos(note.start))
            encoding.append([info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                            1 if inst.is_drum else note.pitch, duration, v2e(note.velocity), info[1], info[3]])
    if len(encoding) == 0:
        return list()

    encoding.sort()
    encoding, is_major, pitch_shift = normalize_to_c_major(encoding)


    # extract chords
    if with_chord:
        max_pos = 0
        note_items = []
        for note in encoding:
            if 0 < note[3] < 128: # and str(note[2]) in condition_inst:
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
        if max_chords > 0:
            chords = infer_chords_for_sequence(note_items,
                                        pos_per_chord=pos_per_chord,
                                        max_chords=max_chords,
                                        key_chord_loglik=key_chord_loglik,
                                        key_chord_transition_loglik=key_chord_transition_loglik
                                        )
        else:
            chords = []    
            
        bar_idx = 0
        for c in chords:
            if c == 'N.C.':
                bar_idx+=1
                continue
            r, k = c.split(':')
            if k == '':
                k = 'null'
            elif k == '7':
                k = 'seven'
            encoding.append((bar_idx, 0, 129, root_dict[r], kind_dict[k], 0, t2e(time_signature_reduce(4, 4)), 0))
            bar_idx += 1

        encoding.sort()

    return encoding, pitch_shift, tpc

def encoding_to_MIDI(encoding, tpc, decode_chord):

    tmp = encoding.strip().split('<0-')[1:]

    encoding = []
    for item in tmp:
        tmp2 = item.strip()[:-1].split('> <')
        encoding.append([int(tmp2[0])] + [int(i[2:]) for i in tmp2[1:]])
    del tmp
    
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
    midi_obj.tempo_changes = tpc

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.midi.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]

    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]

        if program == 129 and decode_chord:
            root_name = root_list[i[3]]
            kind_name = kind_list[i[4]]
            root_pitch_shift = root_dict[root_name]
            end = start + get_tick(0, e2d(1))
            for kind_shift in _CHORD_KIND_PITCHES[kind_name]:
                pitch = 36 + root_pitch_shift + kind_shift
                midi_obj.instruments[1].notes.append(miditoolkit.midi.containers.Note(
                start=start, end=end, pitch=pitch, velocity=e2v(20)))
        elif program != 129:
            pitch = (i[3] - 128 if program == 128 else i[3])
            if pitch < 0:
                continue
            duration = get_tick(0, e2d(i[4]))
            if duration == 0:
                duration = 1
            end = start + duration
            velocity = e2v(i[5])

            midi_obj.instruments[program].notes.append(miditoolkit.midi.containers.Note(
                start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.midi.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.midi.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj

def create_pos_from_str(str_cmd, pos):
    print(str_cmd)
    if str_cmd == '-':
        return pos
    track_cmds = str_cmd.split(';')
    for track_cmd in track_cmds:
        track_id, start_pos, end_pos = track_cmd.split(',')
        if end_pos != '':
            pos[int(track_id) * 2][int(start_pos):int(end_pos)] = 1
            pos[int(track_id) * 2 + 1][int(start_pos):int(end_pos)] = 1
        else:
            pos[int(track_id) * 2][int(start_pos):] = 1
            pos[int(track_id) * 2 + 1][int(start_pos):] = 1
    return pos

def F(file_name):
    
    global tokens_to_ids
    global ids_to_tokens
    global empty_index
    global pad_index

    midi_obj = miditoolkit.midi.parser.MidiFile(file_name)

    encoding, pitch_shift, tpc = MIDI_to_encoding(midi_obj)

    if len(encoding) == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None, 0

    bar_index_offset = 0

    figure_size = max(encoding[-1][0] * pos_in_bar + encoding[-1][1], 512)

    pad_length = 1 #(512 - figure_size % 512)

    figure_size += pad_length

    datum = pad_index * torch.ones(14, figure_size, dtype=float)
    
    oov = 0
    inv = 0
    
    chord_list = []
    
    tempo = b2e(67)

    lead_start = 0

    idx = 0

    track_set = set()

    while idx != len(encoding) - 1:
        e = encoding[idx]

        bar = e[0]
        pos = e[1]
        inst = e[2]
        pitch = e[3]

        if inst == 80:
            tempo = e[7]
            assert tempo != 0, 'bad tempo'
        
        # assert e[6] == 6
        
        if e[2] == 129:
            row = inst_to_row[str(inst)]
            r = root_list[e[3]]
            k = kind_list[e[4]]
            datum[2 * row][pos_in_bar * bar + pos : pos_in_bar * (bar + 1) + pos] = tokens_to_ids[r]
            datum[2 * row + 1][pos_in_bar * bar + pos : pos_in_bar * (bar + 1) + pos] = tokens_to_ids[k]
            idx += 1
            continue
        
        chord_list = [str(e[3])]

        for f_idx in range(idx + 1, len(encoding)):
            if (encoding[f_idx][0] == bar) and (encoding[f_idx][1] == pos) and (encoding[f_idx][2] == inst):
                if encoding[f_idx][3] != pitch:
                    chord_list.append(str(encoding[f_idx][3]))
                    pitch = encoding[f_idx][3]
            else:
                break
        
        idx = max(idx + 1, f_idx)
        
                
        dur = e[4]
        if dur == 0:
            continue
        
        if not (str(inst) in inst_to_row):
            continue
        
        row = inst_to_row[str(inst)]
        dur = tokens_to_ids['T'+str(e[4])] # duration
        
        chord_string = ' '.join(chord_list)
        token = prog_to_abrv[str(inst)] + chord_string

        track_set.add(track_name[prog_to_abrv[str(inst)]])

        if token in tokens_to_ids:
            pitch = tokens_to_ids[token]
            assert (dur < pad_index) and (pitch > pad_index), 'pitch index is {} and dur index is {}'.format(pitch, dur)
            datum[2 * row][pos_in_bar * bar + pos] = pitch
            datum[2 * row + 1][pos_in_bar * bar + pos] = dur
            inv += 1
        else:
            oov += 1

    datum[:,-pad_length:] = empty_index

    print('The music has {} tracks, with {} positions'.format(track_set, datum.size()[1]))
    print('Representation Visualization:')
    print('\t0,1,2,3,4,5,6,7,8,...\n(0)lead\n(1)bass\n(2)drum\n(3)guitar\n(4)piano\n(5)string\n(6)chord')
    print('Example: condition on 100 to 200 position of lead, 300 to 400 position of piano, write command like this:\'0,100,200;4,300,400')
    condition_str = input('Input positions you want to condition on:')
    empty_str = input('Input positions you want to empty:')

    empty_pos = torch.zeros_like(datum)
    condition_pos = torch.zeros_like(datum)
    empty_pos = create_pos_from_str(empty_str, empty_pos)
    condition_pos = create_pos_from_str(condition_str, condition_pos)

    datum = torch.where(empty_pos.type(torch.bool), empty_index, datum)
    datum = torch.where(((datum != empty_index).float() * (1 - condition_pos)).type(torch.bool), empty_index + 1, datum)
    
    not_empty_pos = (torch.tensor(np.array(datum)) != empty_index).float()
    
    return datum.unsqueeze(0), torch.tensor(tempo), not_empty_pos, condition_pos, pitch_shift, tpc

def main_worker(local_rank, args):

    args.local_rank = local_rank

    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    if args.debug:
        config = modify_config_for_debug(config)

    logger = Logger(args)

    global tokens_to_ids
    global ids_to_tokens
    global empty_index
    global pad_index

    with open(config['solver']['vocab_path'],'r') as f:
        tokens = f.readlines()

        for id, token in enumerate(tokens):
            token, freq = token.strip().split('\t')
            tokens_to_ids[token] = id
            ids_to_tokens.append(token)
        pad_index = tokens_to_ids['<pad>']
        empty_index = len(ids_to_tokens)

    model = build_model(config, args)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    dataloader_info = None

    solver = Solver(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger, is_sample=True)

    assert args.load_path is not None
    solver.resume(path=args.load_path)

    file_list = [os.path.join(args.file_path, n) for n in os.listdir(args.file_path) if (n[-4:].lower() == '.mid' or n[-5:].lower() == '.midi') and ('iter' not in n.lower())]
    file_list.sort()

    for file_name in file_list:
        print(file_name)

        if '.pth' in file_name:
            continue
        y = input('skip?')
        if 'y' in y:
            continue

        x, tempo, not_empty_pos, condition_pos, pitch_shift, tpc = F(file_name)

        oct_line = solver.infer_sample(x, tempo, not_empty_pos, condition_pos, use_ema=args.no_ema, skip_step=args.skip_step)
        
        data = oct_line.split(' ')

        oct_final_list = []
        for start in range(3, len(data),8):
            if 'pad' not in data[start] and 'pad' not in data[start+1]:
                pitch = int(data[start][:-1].split('-')[1])
                if data[start-1] != '<2-129>' and data[start-1] != '<2-128>':
                    pitch -= pitch_shift
                data[start] = '<3-{}>'.format(pitch) # re-normalize            
                oct_final_list.append(' '.join(data[start-3:start+5]))
        
        oct_final = ' '.join(oct_final_list)

        midi_obj = encoding_to_MIDI(oct_final, tpc, args.decode_chord)

        save_path = os.path.join(args.file_path, 'position-{}'.format(file_name.split('/')[-1]))

        midi_obj.dump(save_path)    

def main():
    args = get_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        print('args.num_node ', args.num_node)
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
       
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node # 
    
    launch(main_worker, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))


if __name__ == '__main__':
    main()
