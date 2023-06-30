# Author: Botao Yu

import os
import numpy as np
import pickle

from .tools.midi_utils import track_separate as tc
from .tools.midi_utils.filter_and_merge import predict_track_with_model


cur_path = os.path.dirname(__file__)
midi_model_dir = os.path.join(cur_path, 'tools/midi_model')
melody_model = pickle.load(open(os.path.join(midi_model_dir, 'melody_model_new'), 'rb'))
bass_model = pickle.load(open(os.path.join(midi_model_dir, 'bass_model'), 'rb'))
chord_model = pickle.load(open(os.path.join(midi_model_dir, 'chord_model'), 'rb'))


def extract_main_melody_for_file_path(path):
    pm, melody_tracks_idx, bass_tracks_idx = predict_track_with_model(path, melody_model, bass_model, chord_model)

    if len(melody_tracks_idx) == 0:
        return None

    if melody_tracks_idx.shape[0] > 1:
        average_pitch = []
        for instrument in melody_tracks_idx:
            pitch = []
            for note in pm.instruments[instrument].notes:
                pitch.append(note.pitch)
            pitch = np.array(pitch)
            average_pitch.append(np.mean(pitch))
        max_avg_idx = average_pitch.index(max(average_pitch))
        melody_idx = melody_tracks_idx[max_avg_idx]
    elif melody_tracks_idx.shape[0] == 1:
        melody_idx = melody_tracks_idx[0]
    else:
        melody_idx = None

    return melody_idx
