import json
import multiprocessing
import random
import sys
sys.path.append("..")
import miditoolkit
import numpy.random
import pandas as pd
import os, collections, pickle, shutil
import numpy as np
from MidiProcessor.midiprocessor import MidiEncoder, midi_utils, MidiDecoder
from tqdm import tqdm
import gc
from multiprocessing import Pool, Manager, Lock
import math, random
from typing import List, Dict
from functools import partial
from jSymbolic_lib.jSymbolic_util import read_all_feature
from sklearn.preprocessing import StandardScaler
import joblib

random.seed(42)
np.random.seed(42)


def binarize_data(path_root):
    save_root = path_root + "/data-bin"
    dict_path = save_root + f"/dict.txt"
    command = f"fairseq-preprocess --only-source --destdir {save_root} --srcdict {dict_path} "\
              f"--validpref {path_root}/valid.txt  --testpref {path_root}/test.txt  --trainpref {path_root}/train.txt --workers 4 "
    text = os.popen(command).read()
    print(text)


def binarize_command(command, thresholds):
    discrete_feature = []
    for k in range(command.shape[0]):
        thres = thresholds[k]
        discrete_feature.append(np.searchsorted(thres, command[k]))
    return discrete_feature

def gen_split_data(path_root):
    feature_index = np.load("../data/feature_index.npy", allow_pickle=True)
    thresholds = np.load("../data/threshold.npy", allow_pickle=True)
    save_root = path_root
    os.makedirs(save_root, exist_ok=True)
    fn_list = os.listdir(path_root + "/remi")
    random.shuffle(fn_list)
    for split in ["train", "valid", "test"]:
        split_command = []
        if split == "train":
            s, e = 0, int(len(fn_list)*0.8)
        elif split == "valid":
            s,e = int(len(fn_list)*0.8), int(len(fn_list)*0.9)
        else:
            s,e = int(len(fn_list)*0.9), len(fn_list)

        with open(path_root + f"/{split}.txt", "w") as split_txt:
            split_fn_list = []
            j = 0
            for i, fn in enumerate(tqdm(fn_list[s:e])):
                fn_name = fn.split(".")[0]
                try:
                    jS_feature = read_all_feature(path_root + f"/feature/{fn_name}.xml")
                except:
                    continue
                jS_feature = np.array(jS_feature)
                if len(jS_feature) != 1495:
                    continue
                jS_feature = np.array(jS_feature)
                binary_command = binarize_command(jS_feature[feature_index], thresholds)
                split_command.append(binary_command)
                split_fn_list.append(fn)
                with open(path_root + f'/remi/{fn}', "r") as f:
                    remi_tokens = f.read().strip("\n").strip(" ")
                    split_txt.write(remi_tokens + "\n")
                j += 1
            split_command = np.array(split_command)
            np.save(save_root + f"/{split}_fn_list.npy", split_fn_list)
            np.save(save_root + f'/{split}_command.npy', split_command)
            assert len(split_fn_list) == len(split_command), "length dismatch!"

if __name__ == "__main__":
    gen_split_data("../data/Piano")
    binarize_data("../data/Piano")
