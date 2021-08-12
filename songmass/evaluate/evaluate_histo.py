# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#


import os
import io
import shutil
import argparse

import numpy as np


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


def get_pitch_duration_sequence(notes):
    seq = []

    i = 0
    while i < len(notes):
        if notes[i] > 128:
            i += 1
        else:
            if i + 1 >= len(notes):
                break
            if notes[i + 1] <= 128:
                i += 1
            else:
                pitch = str(notes[i])
                duration = str(notes[i + 1])

                seq.extend([pitch, duration])
                i += 2
    return seq



def separate_sentences(x, find_structure=False, SEP='[sep]'):
    z = x.copy()
    separate_positions = [k for k, v in enumerate(z) if v == SEP]
    separate_positions.insert(0, -1)

    sents = []
    for i in range(len(separate_positions) - 1):
        u, v = separate_positions[i] + 1, separate_positions[i + 1]
        sent = z[u:v]
        if find_structure:
            sent = list(map(int, sent))
            sent = get_pitch_duration_sequence(sent)
        sents.append(sent)
    return sents


def get_lyrics(lyric_file):
    with open(lyric_file, 'r') as input_file:
        lines = input_file.readlines()
    lyrics = list(map(lambda x : x.rstrip('\n').split(' '), lines))
    return lyrics


def get_song_ids(song_id_file):
    with open(song_id_file, 'r') as input_file:
        song_ids = input_file.readlines()
    song_ids = list(map(lambda x : int(x.rstrip('\n')), song_ids))
    return song_ids


def get_songs(
    melody_file,
    lyric_file=None,
    song_id_file=None,
    is_generated=False, 
    get_last=False,
    find_structure=False,
    cut_exceed_sent=False,
    beam=5,
    SEP='[sep]',
    ALIGN='[align]',
):

    lyrics = get_lyrics(lyric_file)
    song_ids = get_song_ids(song_id_file)
    lyric_sents = list(map(lambda x : x.count(SEP), lyrics))

    def to_tuple(x):
        pitch_duration = [i for i in x if i != SEP and i != ALIGN]
        pd_tuples = [(pitch_duration[2 * i], pitch_duration[2 * i + 1]) for i in range(len(pitch_duration) // 2)]
        return pd_tuples
    
    with open(melody_file, 'r') as input_file:
        melodies = input_file.readlines()

        if is_generated:
            melodies = list(filter(lambda x : x.startswith('H-'), melodies))
            if len(melodies) == len(lyrics) * beam:
                melodies.sort(key = lambda x : (int(x.split('\t')[0].split('-')[1]), - float(x.split('\t')[1])))
                melodies = [x for i, x in enumerate(melodies) if i % beam == 0]
            else:
                melodies.sort(key = lambda x : int(x.split('\t')[0].split('-')[1]))

    melodies = list(map(lambda x : x.rstrip('\n').split('\t')[-1], melodies))
    assert len(melodies) == len(lyrics)
    
    melody_seqs = list(map(lambda x : x.rstrip('\n').split(' '), melodies))
    melody_seqs = [i for i in melody_seqs if i != ALIGN]
    
    for i in range(len(melody_seqs)):
        melody_seqs[i] = list(filter(lambda x : x.isdigit() or x == SEP, melody_seqs[i]))

    if get_last:
        for i in range(len(melody_seqs)):
            if melody_seqs[i][-1] != SEP:
                melody_seqs[i].append(SEP)
    
    melody_seq_sents = list(map(lambda x : separate_sentences(x, find_structure=find_structure), melody_seqs))
    
    song_seqs = []
    for i, seq in enumerate(melody_seq_sents):
        if cut_exceed_sent and len(seq) > lyric_sents[i]:
            seq = seq[0 : lyric_sents[i]] 

        song_seq = []
        for k, sent in enumerate(seq):
            song_seq.extend(sent)
            song_seq.append(SEP)
        
        song_seqs.append(song_seq)
        
    song_num = song_ids[-1] + 1
    songs = [[] for _ in range(song_num)]

    for k, v in enumerate(song_ids):
        songs[v].extend(song_seqs[k])
    songs = list(map(to_tuple, songs))
    return songs


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
