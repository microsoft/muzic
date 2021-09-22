import os
import subprocess
from multiprocessing.pool import Pool

import miditoolkit
import pandas as pd
import pretty_midi
from tqdm import tqdm
import numpy as np
import pickle
from copy import deepcopy

from midi_preprocess.utils.hparams import hparams
import midi_preprocess.steps.track_separate as tc


def filter_and_merge(processed_data_dir, instru2program):
    base_dir = 'midi_preprocess'
    melody_model = pickle.load(open(f'{base_dir}/model/melody_model_new', 'rb'))
    bass_model = pickle.load(open(f'{base_dir}/model/bass_model', 'rb'))
    chord_model = pickle.load(open(f'{base_dir}/model/chord_model', 'rb'))

    df = pd.read_csv(open(f'{processed_data_dir}/meta.csv'))
    print(f"| load #midi infos: {df.shape[0]}.")
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    save_dir = f'{processed_data_dir}/midi_recog_tracks'
    subprocess.check_call(f'rm -rf "{save_dir}"', shell=True)
    futures = [pool.apply_async(filter_recog_merge_job, args=[
        midi_info['path'], midi_info, instru2program, save_dir, melody_model, bass_model, chord_model
    ]) for idx, midi_info in df.iterrows()]
    pool.close()
    merged_infos = []
    for f, (idx, midi_info) in zip(tqdm(futures), df.iterrows()):
        res = f.get()
        merged_info = {}
        merged_info.update(midi_info)
        if isinstance(res, str):
            merged_info['msg'] = res
        else:
            merged_info['msg'] = ''
            merged_info.update(res)
        merged_infos.append(merged_info)
    df = pd.DataFrame(merged_infos)
    df = df.set_index(['id'])
    df.to_csv(f'{processed_data_dir}/meta.csv')
    pool.join()
    n_merged = len([x for x in merged_infos if x['msg'] == ''])
    print(f"| merged #midi: {n_merged}")


def predict_track_with_model(midi_path, melody_model, bass_model, chord_model):
    try:
        ret = tc.cal_file_features(midi_path)  # remove empty track and calculate the features
        features, pm = ret
    except Exception as e:
        features = None
        pm = pretty_midi.PrettyMIDI(midi_path)
    if features is None and pm is None:
        pm = pretty_midi.PrettyMIDI(midi_path)
    if features is None or features.shape[0] == 0:
        return pm, [], []
    features = tc.add_labels(features)  # add label
    tc.remove_file_duplicate_tracks(features, pm)  # delete duplicate track
    features = tc.predict_labels(features, melody_model, bass_model, chord_model)  # predict lead, bass, chord
    predicted_melody_tracks_idx = np.where(features.melody_predict)[0]
    predicted_bass_tracks_idx = np.where(features.bass_predict)[0]
    melody_tracks_idx = np.concatenate((predicted_melody_tracks_idx, np.where(features.is_melody)[0]))
    bass_tracks_idx = np.concatenate((predicted_bass_tracks_idx, np.where(features.is_bass)[0]))
    return pm, melody_tracks_idx, bass_tracks_idx


def filter_recog_merge_job(midi_path, midi_info, instru2program, save_dir,
                           melody_model, bass_model, chord_model):
    filter_msg = filter_tracks(midi_info)
    if filter_msg != '':
        return filter_msg
    pm, melody_tracks_idx, bass_tracks_idx = predict_track_with_model(midi_path, melody_model, bass_model, chord_model)
    if pm is None:
        return 'pm is None'
    pm_new = deepcopy(pm)
    pm_new.instruments = []
    for i, instru_old in enumerate(pm.instruments):
        program_old = instru_old.program
        instru = deepcopy(instru_old)

        if i in melody_tracks_idx and 'MUMIDI_' not in instru.name or instru.name == 'MUMIDI_Lead':
            instru.name = 'Lead'
        elif i in bass_tracks_idx and 'MUMIDI_' not in instru.name or instru.name == 'MUMIDI_Bass':
            instru.name = 'Bass'
        elif instru_old.is_drum and 'MUMIDI_' not in instru.name or instru.name == 'MUMIDI_Drums':  # drum
            instru.name = 'Drums'
        elif program_old // 8 == 0 and 'MUMIDI_' not in instru.name or instru.name == 'MUMIDI_Piano':  # piano
            instru.name = 'Piano'
        elif program_old // 8 == 3 and 'MUMIDI_' not in instru.name or instru.name == 'MUMIDI_Guitar':  # guitar
            instru.name = 'Guitar'
        elif 40 <= program_old <= 54 and 'MUMIDI_' not in instru.name or instru.name == 'MUMIDI_Strings':  # string
            instru.name = 'Strings'
        elif 73 <= program_old <= 88:  # Lead
            instru.name = 'Lead'
        elif program_old // 8 == 4:  # Bass
            instru.name = 'Bass'
        else:
            instru.name = 'UnRec'
        instru.program = instru_old.program
        pm_new.instruments.append(instru)
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/{midi_info['id']}.mid"
    pm_new.write(out_path)
    merged_midi_info = get_merged_midi_info(out_path, instru2program)
    filter_msg = filter_tracks(midi_info)
    if filter_msg != '':
        return '[merged]' + filter_msg
    return merged_midi_info


def filter_tracks(midi_info):
    # filter out too long n_beats > 10000, and too short n_beats < 16
    if midi_info['n_beats'] > hparams['max_n_beats'] or midi_info['n_beats'] < hparams['min_n_beats']:
        return 'invalid beats'

    if midi_info['n_notes'] < hparams['min_n_notes']:
        return 'invalid n_notes'

    if midi_info['n_pitches'] < hparams['min_n_pitches']:
        return 'Invalid pitches'

    if midi_info['cross_bar_rate'] > hparams['max_cross_bar_rate']:
        return 'Invalid cross_bar'

    return ''


def get_merged_midi_info(midi_fn, instru2program):
    try:
        mf = miditoolkit.MidiFile(midi_fn)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return str(e)

    # merge tracks
    track_lists_to_merge = get_tracks_to_merge(mf, instru2program)

    n_merge_track = [len(x) for x in track_lists_to_merge]
    available_instrs = list(set([x2 for x in track_lists_to_merge for x2 in x]))  # Important for 6 tracks

    # notes
    all_vels = [x1.velocity for i, x in enumerate(mf.instruments) if i in available_instrs for x1 in
                x.notes]  # all instruments note connection in a line
    all_pitches = [x1.pitch for i, x in enumerate(mf.instruments) if i in available_instrs for x1 in x.notes]
    n_notes = len(all_vels)  # numbers of notes

    if n_notes == 0:
        return 'empty tracks'

    n_beats = max([x1.end for i, x in enumerate(mf.instruments)
                   if i in available_instrs for x1 in x.notes]) // mf.ticks_per_beat + 1

    n_instru = len(mf.instruments)
    n_pitches = len(set(all_pitches))  # pitch classes
    vel_mean = np.mean(all_vels)
    vel_std = np.std(all_vels)

    n_cross_bar = 0
    for i, instru in enumerate(mf.instruments):
        if i not in available_instrs:
            continue
        for n in instru.notes:
            start_beat = n.start / mf.ticks_per_beat
            end_beat = n.end / mf.ticks_per_beat
            if (start_beat + 0.25) // 4 < (end_beat - 0.25) // 4 and start_beat % 4 > 0.125:
                n_cross_bar += 1

    return {
        'path_recog_tracks': midi_fn,
        # velocity
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        # stats
        'n_notes': n_notes,
        'n_instru': n_instru,
        'n_beats': n_beats,
        'n_pitches': n_pitches,
        'n_cross_bar': n_cross_bar,
        # tracks
        'n_tracks': n_merge_track,
        'track_lists_to_merge': track_lists_to_merge,
    }


def get_tracks_to_merge(mf, instru2program):
    track_lists_to_merge = [[] for _ in range(6)]
    instru_order = {v: k for k, v in enumerate(instru2program.keys())}
    for idx, instr in enumerate(mf.instruments):
        instr_name = instr.name
        if instr_name in instru_order:
            track_lists_to_merge[instru_order[instr_name]].append(idx)
    return track_lists_to_merge