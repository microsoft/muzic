# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import tables
import sys
import random
import zipfile
from multiprocessing import Pool, Manager
import preprocess
import json
from sklearn.model_selection import StratifiedKFold


#subset = input('subset: ')
subset = "hotness"
raw_data_dir = subset + '_data_raw'
if os.path.exists(raw_data_dir):
    print('Output path {} already exists!'.format(raw_data_dir))
    #sys.exit(0)

#data_path = input('LMD dataset zip path: ')
data_path = "lmd_matched.zip"
n_folds = 5
n_times = 4  # sample train set multiple times
#max_length = int(input('sequence length: '))
max_length = 1000
preprocess.sample_len_max = max_length
preprocess.deduplicate = False
preprocess.data_zip = zipfile.ZipFile(data_path)
fold_map = dict()
manager = Manager()
all_data = manager.list()
pool_num = 24

RESULTS_PATH = ""
# Utility functions for retrieving paths
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def msd_id_to_h5(msd_id):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join(RESULTS_PATH, 'lmd_matched_h5',
                        msd_id_to_dirs(msd_id) + '.h5')

def get_midi_path(msd_id, midi_md5, kind):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(RESULTS_PATH, 'lmd_{}'.format(kind),
                        msd_id_to_dirs(msd_id), midi_md5 + '.mid')

# TODO this is what we'll replace with the hotness score
# Changes
# 1. Doing single label (for the output)
# 2. Read from the aligned

labels = dict()
#with open('midi_genre_map.json') as f:
#    for s in json.load(f)[subset].items():
#        labels[s[0]] = tuple(
#            sorted(set(i.strip().replace(' ', '-') for i in s[1])))


def get_id(file_name):
    return file_name.split('/')[-1].split('.')[0]


def get_fold(file_name):
    return fold_map[get_id(file_name)]


def get_sample(output_str_list):
    max_len = max(len(s.split()) for s in output_str_list)
    return random.choice([s for s in output_str_list if len(s.split()) == max_len])


def new_writer(file_name, output_str_list):
    if len(output_str_list) > 0:
        all_data.append((file_name, tuple(get_sample(output_str_list)
                                          for _ in range(n_times))))


preprocess.writer = new_writer


os.system('mkdir -p {}'.format(raw_data_dir))
file_list = [file_name for file_name in preprocess.data_zip.namelist(
) if file_name[-4:].lower() == '.mid' or file_name[-5:].lower() == '.midi']
#file_list = [file_name for file_name in file_list if get_id(
#    file_name) in labels]
file_list = file_list[:100] # TODO just testing
for file_name in file_list:
    #print(file_name)
    msd_id = file_name.split("/")[-2]#[:-4]
    #print(msd_id)
    #h5 = file_name.replace("mid", "h5")
    #h5 = h5.replace("lmd_matched", "lmd_matched_h5")
    #print(file_name)
    #print(get_id(file_name))
    #h5 = h5.replace(file_name, get_id(file_name))
    #print(file_name)
    #msd_id = get_id(file_name)
    #print('ID: {}'.format(msd_id))
    #print('ID: {}'.format(msd_id_to_h5(msd_id)))
    print(msd_id_to_dirs(msd_id))
    with tables.open_file(msd_id_to_h5(msd_id)) as h5:
        print("filename {} hotness {}".format(file_name, h5.root.metadata.songs.cols.song_hotttnesss[0]))
        #print(dir(h5.root.metadata.songs.cols))
        print('"{}" by {} on "{}"'.format(
            h5.root.metadata.songs.cols.title[0],
            h5.root.metadata.songs.cols.artist_name[0],
            h5.root.metadata.songs.cols.release[0]))
        print("\n\n")

raise ValueError("DONE")

random.shuffle(file_list)
label_list = ['+'.join(labels[get_id(file_name)]) for file_name in file_list]
fold_index = 0
for train_index, test_index in StratifiedKFold(n_folds).split(file_list, label_list):
    for i in test_index:
        fold_map[get_id(file_list[i])] = fold_index
    fold_index += 1
with Pool(pool_num) as p:
    list(p.imap_unordered(preprocess.G, file_list))
random.shuffle(all_data)
print('{}/{} ({:.2f}%)'.format(len(all_data),
                               len(file_list), len(all_data) / len(file_list) * 100))
for fold in range(n_folds):
    os.system('mkdir -p {}/{}'.format(raw_data_dir, fold))
    preprocess.gen_dictionary('{}/{}/dict.txt'.format(raw_data_dir, fold))
    for cur_split in ['train', 'test']:
        output_path_prefix = '{}/{}/{}'.format(raw_data_dir, fold, cur_split)
        with open(output_path_prefix + '.txt', 'w') as f_txt:
            with open(output_path_prefix + '.label', 'w') as f_label:
                with open(output_path_prefix + '.id', 'w') as f_id:
                    count = 0
                    for file_name, output_str_list in all_data:
                        if (cur_split == 'train' and fold != get_fold(file_name)) or (cur_split == 'test' and fold == get_fold(file_name)):
                            for i in range(n_times if cur_split == 'train' else 1):
                                f_txt.write(output_str_list[i] + '\n')
                                f_label.write(
                                    ' '.join(labels[get_id(file_name)]) + '\n')
                                f_id.write(get_id(file_name) + '\n')
                                print("Sample Text Data:", output_str_list[i])
                                print("Sample Label:", ' '.join(labels[get_id(file_name)]))
                                print("Sample ID:", get_id(file_name))
                                count += 1
                    print(fold, cur_split, count)
