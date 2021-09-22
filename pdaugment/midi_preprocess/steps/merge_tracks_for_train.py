import os
import subprocess
from multiprocessing.pool import Pool
import pandas as pd
from tqdm import tqdm
import miditoolkit
import numpy as np

from midi_preprocess.utils.hparams import hparams
from utils.midi_chord import infer_chords_for_midi
from .merge_track_ops import keep_long_notes, keep_track_with_most_notes, merge_strings, merge_lead

STEP_PER_BAR = 32  # steps per bar
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]
DEFAULT_TICKS_PER_BAR = 480
TICKS_PER_STEP = DEFAULT_TICKS_PER_BAR * 4 // STEP_PER_BAR


def gen_merge_midi(processed_data_dir, step_per_bar, track_info, instru2track):
    df = pd.read_csv(f'{processed_data_dir}/meta.csv')
    df = df[~pd.isnull(df['path_recog_tracks'])]
    print(f"| load #midi info: {len(df)}")
    save_dir = f'{processed_data_dir}/midi_6tracks'
    subprocess.check_call(f'rm -rf "{save_dir}"', shell=True)
    os.makedirs(save_dir, exist_ok=True)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [
        pool.apply_async(merge_6tracks, args=[
            midi_info['path_recog_tracks'], midi_info, step_per_bar, track_info,
            instru2track, save_dir
        ]) for idx, midi_info in df.iterrows()]
    pool.close()
    results = [x.get() for x in tqdm(futures)]
    n_merged = len([x for x in results if x == ''])
    pool.join()
    print(f"| merged #midi: {n_merged}.")


def merge_6tracks(midi_path, midi_info, step_per_bar, track_info, instru2track, save_dir):
    mf = miditoolkit.MidiFile(midi_path)
    track_lists_to_merge = eval(midi_info['track_lists_to_merge'])
    tick_per_step = mf.ticks_per_beat * 4 / step_per_bar
    midi_info_ts = eval(midi_info['ts'])
    if len(midi_info_ts) != 0:
        ts_set = set(x[1] for x in midi_info_ts)
        if '4/4' not in ts_set:
            return 'no 4/4'
        if len(ts_set) > 0 and any([
            x[0] % mf.ticks_per_beat != 0 for x in midi_info_ts if x[1] == '4/4']):
            return 'no 4/4'

    for instru in mf.instruments:
        vels_std = np.std([n.velocity for n in instru.notes])
        if vels_std == 0:
            for n in instru.notes:
                n.velocity = 127
        for n in instru.notes:
            n.start = int(round(n.start / tick_per_step) * tick_per_step)
            n.end = int(round(n.end / tick_per_step) * tick_per_step)
            if n.start >= n.end:
                n.end = int(n.end + tick_per_step)
        instru.notes.sort(key=lambda x: (x.start, x.pitch, -x.end))

    new_instrs = []
    for idx, ((name, new_program_id), track_to_merge) in enumerate(zip(track_info.items(), track_lists_to_merge)):
        if len(track_to_merge) == 0:
            continue
        if name in ['Drums', 'Piano']:
            new_instrs.append(keep_long_notes(mf, new_program_id, track_to_merge, name=name))
        elif name in ['Bass', 'Guitar']:
            new_instrs.append(keep_track_with_most_notes(mf, new_program_id, track_to_merge, name=name))
        elif name in ['Lead']:
            new_instrs.append(merge_lead(mf, new_program_id, track_to_merge))
        elif name in ['Strings']:
            new_instrs.append(merge_strings(mf, new_program_id, track_to_merge))
    mf.instruments = [x for x in new_instrs if x is not None]

    tmp_fn = f"{save_dir}/{midi_info['id']}.mid"
    mf.dump(tmp_fn)
    mf = miditoolkit.MidiFile(tmp_fn)
    subprocess.check_call(f'rm -rf "{tmp_fn}"', shell=True)
    mf = infer_chords_for_midi(mf, instru2track=instru2track)
    if len(midi_info_ts) == 0:
        midi_info_ts = [0, '4/4']
    ts = []
    for t in midi_info_ts:
        t = list(t)
        if len(ts) == 0:
            ts.append(t)
        else:
            if t[1] != ts[-1][1]:
                ts[-1].append(t[0])
                ts.append(t)
    ts[-1].append(mf.max_tick)
    for seg_idx, seg in enumerate(ts):
        if seg[1] == '4/4':
            if (seg[2] - seg[0]) / mf.ticks_per_beat > hparams['min_n_beats']:
                mf.tempo_changes = [miditoolkit.TempoChange(midi_info['tempo_mean'], seg[0])]
                mf.dump(f"{save_dir}/{midi_info['id']}_TS{seg_idx}.mid", [seg[0], seg[2]])
    return ''
