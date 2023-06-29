import os
import random
import shutil
import json
from multiprocessing import Process, Pool
from functools import partial
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("..")
from MidiProcessor.midiprocessor import MidiEncoder, MidiDecoder, enc_remigen_utils, const
from MidiProcessor.midiprocessor.keys_normalization import get_notes_from_pos_info, get_pitch_shift
import math, pickle,miditoolkit
import json

random.seed(2022)
np.random.seed(2022)


encoder = MidiEncoder("REMIGEN2")


def midi_encoding(midi_path, save_root, prefix):
    try:
        name_split = midi_path.replace("//", "/").replace("\\", "/").split("/")
        midi_name = name_split[-1][:-4]  # skip ".mid"
        # save_name = "_".join(name_split[-3:-1]) + "_" + f"{midi_name}.txt"
        midi_name = midi_name.replace(" ", "_")
        if prefix is not None:
            save_name = f"{prefix}_{midi_name}.txt"
        else:
            save_name = f"{midi_name}.txt"
        midi_obj = miditoolkit.MidiFile(midi_path)
        remi_token = encoder.encode_file(file_path = midi_path, midi_obj = midi_obj, remove_empty_bars=True)
        encoder.dump_token_lists(token_lists=remi_token, file_path=os.path.join(save_root, save_name))
        return 1
    except KeyboardInterrupt:
        sys.exit(1)
    except BaseException:
        print(midi_path, "error")
        return 0

def midi_encoding_generate(data_path):

    midi_path_list = os.listdir(data_path + f"/midi")
    for i in range(len(midi_path_list)):
        midi_path_list[i] = data_path + f"/midi/{midi_path_list[i]}"

    save_root = data_path + f"/remi"
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    with Pool(processes=8) as pool:
        result = iter(tqdm(pool.imap(partial(midi_encoding, save_root = save_root, prefix = None), midi_path_list), total=len(midi_path_list)))
        for i in range(len(midi_path_list)):
            # try:
                next(result)
            # except BaseException as e:
            #     if isinstance(e, KeyboardInterrupt):
            #         print("error")
            #         pool.terminate()
            #     else:
            #         print(midi_path_list[i], "error!")



if __name__ == "__main__":
    midi_encoding_generate("../data/Piano")

