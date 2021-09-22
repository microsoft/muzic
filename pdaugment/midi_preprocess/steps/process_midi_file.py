import os
import glob
import subprocess
import traceback
from multiprocessing.pool import Pool

import miditoolkit
import mido
from tqdm import tqdm
import pretty_midi
import numpy as np
import pandas as pd

DEFAULT_TICKS_PER_BAR = 480


def process_midi_file(raw_data_dir, processed_data_dir, hparams):
    if not isinstance(raw_data_dir, list):
        raw_data_dir = [raw_data_dir]
    os.makedirs(processed_data_dir, exist_ok=True)
    dir_name = f'{processed_data_dir}/midi'
    subprocess.check_call(f'rm -rf "{dir_name}"', shell=True)
    os.makedirs(dir_name, exist_ok=True)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = []
    for ds_idx, data_path in enumerate(raw_data_dir):
        midi_fns = sorted(glob.glob(f'{data_path}/**/*.mid*', recursive=True))
        if hparams['max_samples_per_ds'] > 0:
            midi_fns = midi_fns[:hparams['max_samples_per_ds']]
        futures += [pool.apply_async(save_midi, args=[
            midi_fn, f'{ds_idx}_' + midi_fn.lstrip(f'{data_path}').replace("/", "_"), processed_data_dir
        ]) for midi_fn in midi_fns]
    midi_infos = [x.get() for x in tqdm(futures)]
    midi_infos = [x for x in midi_infos if x is not None]
    df = pd.DataFrame(midi_infos)
    df = df.set_index(['id'])
    df.to_csv(f'{processed_data_dir}/meta.csv')

    pool.close()
    pool.join()


def save_midi(midi_file, save_fn, dest_dir):
    try:
        try:
            pm = pretty_midi.PrettyMIDI(midi_file)
        except:
            mido.MidiFile(filename=midi_file, clip=True).save(midi_file)
            pm = pretty_midi.PrettyMIDI(midi_file)
        midi_fn = f'{dest_dir}/midi/{save_fn}'
        pm.write(midi_fn)

        mf = miditoolkit.MidiFile(midi_fn)
        file_resolution = mf.ticks_per_beat
        mf.ticks_per_beat = DEFAULT_TICKS_PER_BAR
        for instru in mf.instruments:
            for n in instru.notes:
                n.start = int(round(n.start / file_resolution * DEFAULT_TICKS_PER_BAR))
                n.end = int(round(n.end / file_resolution * DEFAULT_TICKS_PER_BAR))
        for marker in mf.markers:
            marker.time = int(round(marker.time / file_resolution * DEFAULT_TICKS_PER_BAR))
        for lyric in mf.lyrics:
            lyric.time = int(round(lyric.time / file_resolution * DEFAULT_TICKS_PER_BAR))
        for tempo in mf.tempo_changes:
            tempo.time = int(round(tempo.time / file_resolution * DEFAULT_TICKS_PER_BAR))
        for ks in mf.key_signature_changes:
            ks.time = int(round(ks.time / file_resolution * DEFAULT_TICKS_PER_BAR))
        mf.dump(midi_fn)
        mf = miditoolkit.MidiFile(midi_fn)

        # tempo
        tempo = [(x.time, x.tempo) for x in mf.tempo_changes]
        tempo_mean, tempo_std = -1, -1
        if len(tempo) > 0:
            tempo_mean = np.mean([x[1] for x in tempo])
            tempo_std = np.std([x[1] for x in tempo])

        # time signature
        ts = [(x.time, '{}/{}'.format(x.numerator, x.denominator)) for x in mf.time_signature_changes]
        ts = [x for idx, x in enumerate(ts) if idx == 0 or ts[idx - 1][1] != x[1]]

        # key signature
        ks = [(x.time, x.key_name) for x in mf.key_signature_changes]

        all_vels = [x1.velocity for i, x in enumerate(mf.instruments) for x1 in x.notes]
        all_pitches = [x1.pitch for i, x in enumerate(mf.instruments) for x1 in x.notes]
        n_notes = len(all_vels) 

        if n_notes == 0:
            return None

        n_beats = max([x1.end for i, x in enumerate(mf.instruments) for x1 in x.notes]) // mf.ticks_per_beat + 1

        n_instru = len(mf.instruments)
        n_pitches = len(set(all_pitches)) 
        vel_mean = np.mean(all_vels)
        vel_std = np.std(all_vels)

        id = os.path.basename(midi_fn).replace('.midi', '').replace('.mid', '')
        n_cross_bar = 0
        for i, instru in enumerate(mf.instruments):
            for n in instru.notes:
                start_beat = n.start / mf.ticks_per_beat
                end_beat = n.end / mf.ticks_per_beat
                if (start_beat + 0.25) // 4 < (end_beat - 0.25) // 4 and start_beat % 4 > 0.125:
                    n_cross_bar += 1

        info_dict = {
            'id': id,
            'path': midi_fn,
            'tick_per_beat': mf.ticks_per_beat,
            # time signature
            'ts': ts,
            'n_ts': len(ts),
            # tempos
            'tempos': tempo,
            'n_tempo': len(tempo),
            'tempo_mean': tempo_mean,
            'tempo_std': tempo_std,
            # ks
            'ks': ks,
            'n_ks': len(ks),
            # velocity
            'vel_mean': vel_mean,
            'vel_std': vel_std,
            # stats
            'n_notes': n_notes,
            'n_instru': n_instru,
            'n_beats': n_beats,
            'n_pitches': n_pitches,
            'n_cross_bar': n_cross_bar,
            'cross_bar_rate': n_cross_bar / n_notes
        }
        return info_dict

    except Exception as e:
        # traceback.print_exc()
        print(f"| load data error ({type(e)}: {e}): ", midi_file)
        return None
