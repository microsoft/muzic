#!/usr/bin/env python
# coding: utf-8

import io
import sys
import numpy as np
import scipy.linalg
import os
import shutil

data_dir = sys.argv[1]
input_folder  = 'l2m'
output_folder = 'l2m_merge'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

num_layers = 6

with io.open(f'{data_dir}/para/song_id.txt') as f:
    split_to_songid = f.readlines()
split_to_songid = list(map(lambda x:int(x.rstrip('\n')),split_to_songid))



from collections import defaultdict
songid_to_split = defaultdict(list)



for split_id, song_id in enumerate(split_to_songid):
    songid_to_split[song_id].append(split_id)

print(songid_to_split)

# 歌词/旋律
for song_id in songid_to_split.keys():
    song_folder = os.path.join(output_folder, str(song_id))
    os.makedirs(song_folder)
    all_lyric = []
    all_melody = []
    for split_id in songid_to_split[song_id]:
        with io.open(os.path.join(input_folder, str(split_id), 'train.lyric'), 'r') as f:
            lyric = f.read().rstrip('\n').split(' ')
        all_lyric.extend(lyric)
        with io.open(os.path.join(input_folder, str(split_id), 'train.melody'), 'r') as f:
            melody = f.read().rstrip('\n').split(' ')
        all_melody.extend(melody)
    with io.open(os.path.join(song_folder, 'lyric'), 'w') as f:
        f.write(" ".join(all_lyric))
    with io.open(os.path.join(song_folder, 'melody'), 'w') as f:
        f.write(" ".join(all_melody))



for song_id in songid_to_split.keys():
    song_folder = os.path.join(output_folder, str(song_id))
    attn_arr = None
    for split_id in songid_to_split[song_id]:
        with io.open(os.path.join(input_folder, str(split_id), 'attn.txt'), 'r') as f:
            attn = np.array(eval(f.read().rstrip('\n')))
            if attn_arr is None:
                attn_arr = [attn[i,0,:-1,:-1] for i in range(num_layers)]
            else:
                attn_arr = [scipy.linalg.block_diag(attn_arr[i],attn[i,0,:-1,:-1]) for i in range(num_layers) ]
    attn_arr=np.array(attn_arr)
    print(attn_arr.shape)
    with io.open(os.path.join(song_folder, 'attn'), 'w') as f:
        f.write(str(attn_arr.tolist())) 
