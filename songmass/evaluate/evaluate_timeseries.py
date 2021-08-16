# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#


import os
import io
import shutil
import argparse
import dtw

import numpy as np

from utils import get_songs


parser = argparse.ArgumentParser(description="histogram evaluation for melody.")

parser.add_argument('--lyric-file', type=str,
                    help="The original lyric file")
parser.add_argument('--melody-file', type=str,
                    help="The original melody file")
parser.add_argument('--song-id-file', type=str,
                    help="The original song id file")
parser.add_argument('--generated-melody-file', type=str,
                    help="The generated melody file by")


duration_vocab = dict([
    (str(129 + i), x / 100) for i, x in enumerate(list(range(25, 3325, 25)))
])


def flatten(notes, ignore_rest=False):
    flatten_notes = []

    for note in notes:
        pitch = int(note[0])
        duration = duration_vocab[note[1]]

        if pitch == 128:
            if ignore_rest or len(flatten_notes) == 0:
                continue
            flatten_notes.extend([flatten_notes[-1]] * int(duration * 4))
        else:
            flatten_notes.extend([pitch] * int(duration * 4))

    return flatten_notes


def sample_notes(flatten_notes, freq=2):
    return [flatten_notes[i * freq] for i in range(len(flatten_notes) // freq)]


def main(args):
    hypos = get_songs(
        args.generated_melody_file,
        args.lyric_file,
        args.song_id_file,
        is_generated=True,
        get_last=True,
        find_structure=True,
        cut_exceed_sent=True,
    )

    targets = get_songs(
        args.melody_file,
        args.lyric_file,
        args.song_id_file,
    )

    flatten_hypos   = list(map(flatten, hypos))
    flatten_targets = list(map(flatten, targets))

    flatten_hypo_samples   = list(map(sample_notes, flatten_hypos))
    flatten_target_samples = list(map(sample_notes, flatten_targets))
    
    dtw_mean = []
    for i in range(len(targets)):
        if len(flatten_target_samples[i]) == 0 or len(flatten_hypo_samples[i]) == 0:
            continue

        d1 = np.array(flatten_target_samples[i]).reshape(-1, 1)
        d2 = np.array(flatten_hypo_samples[i]).reshape(-1, 1)

        d1 = d1 - np.mean(d1)
        d2 = d2 - np.mean(d2)
        d, _, _, _ = dtw.accelerated_dtw(d1, d2, dist='euclidean')
        dtw_mean.append(d / len(d2))
    
    print('The melody distance is {}.'.format(sum(dtw_mean) / len(dtw_mean)))
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

