# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#


import os
import io
import shutil
import argparse

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
parser.add_argument('--metric', choices=['pitch', 'duration'], default='pitch',
                    help="Calculate the pitch/duration distribution similarity")


def get_pitch_count(x):
    cnt = [0] * 129
    for i in x:
        cnt[int(i[0])] += 1
    return np.array(cnt)


def get_duration_count(x):
    cnt = [0 for _ in range(25, 3325, 25)]
    for i in x:
        cnt[int(i[1]) - 129] += 1
    return np.array(cnt)


def measure_pitch_simiarlity(targets, hypos):
    song_num = len(hypos)

    similarity = 0

    def get_pitch_histo(x):
        x = get_pitch_count(x)
        x_sum = np.sum(x) if np.sum(x) > 0 else 1
        x_dist = x.astype(np.float32) / x_sum
        return x_dist

    for i in range(song_num):
        hypo_histo = get_pitch_histo(hypos[i])
        target_histo = get_pitch_histo(targets[i])

        pitch_diff = np.abs(hypo_histo - target_histo)
        pitch_overlap_i = (hypo_histo + target_histo - pitch_diff) / 2

        similarity += np.sum(pitch_overlap_i)

    return similarity / song_num


def measure_duration_similarity(targets, hypos):
    song_num = len(hypos)

    similarity = 0

    def get_duration_histo(x):
        x = get_duration_count(x)
        x_sum = np.sum(x) if np.sum(x) > 0 else 1
        x_dist = x.astype(np.float32) / x_sum
        return x_dist

    for i in range(song_num):
        hypo_histo = get_duration_histo(hypos[i])
        target_histo = get_duration_histo(targets[i])

        duration_diff = np.abs(hypo_histo - target_histo)
        duration_overlap_i = (hypo_histo + target_histo - duration_diff) / 2

        similarity += np.sum(duration_overlap_i)

    return similarity / song_num


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

    metric_func = {
        'pitch': measure_pitch_simiarlity,
        'duration': measure_duration_similarity,
    }

    similarity = metric_func[args.metric](targets, hypos)
    print('The {} distribution similarity is {}'.format(args.metric, similarity))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
