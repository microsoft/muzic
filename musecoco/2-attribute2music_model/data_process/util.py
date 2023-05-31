import os

import numpy as np
import random
import shutil
from tqdm import tqdm

key_order = ['I1s1', 'I2s1', 'C1', 'R2', 'S1', 'S2', 'S3', 'B1s1', 'TS1s1', 'K1', 'T1', 'P3', 'P4', 'ST1', 'EM1']
key_has_NA = ["C1", "R2", "S1", "S2", "S3", "B1s1", "TS1s1", "K1", "T1", "ST1", "EM1"]

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

def binarize_data(data_path):
    dict_path = data_path + "/dict.txt"
    command = f"fairseq-preprocess --only-source --destdir {data_path}/data-bin/ --srcdict {dict_path} " \
              f"--validpref {data_path}/valid.txt  --testpref {data_path}/test.txt  --trainpref {data_path}/train.txt --workers 6 "
    text = os.popen(command).read()
    print(text)

if __name__ == "__main__":
    data_path = "../data/truncated_5120"
    binarize_data(data_path)
