import shutil

import pandas as pd
import json
import os
import subprocess
from tqdm import tqdm
import time
import xmltodict
import random
import numpy as np
from multiprocessing import Process, Pool
import json
from jSymbolic_util import read_pitch_feature, read_all_feature
import xmltodict
import subprocess
from functools import partial
import argparse

command_prefix = "java -Xmx6g -jar ./jSymbolic_2_2_user/jSymbolic2.jar -configrun ./jSymbolic_2_2_user/jSymbolicDefaultConfigs.txt"



def rename_midi_path(root):
    midi_name_list = os.listdir(root)
    for midi_name in midi_name_list:
        os.rename(root + "/" + midi_name, root + f"/{midi_name.replace(' ', '_')}")

def get_jSymbolic_feature(file_name, root):
    midi_name = file_name[:-4]

    midi_path = root + f"/midi/{midi_name}.mid"
    path = os.path.join(root, "feature/" + f"{midi_name.replace(' ', '_')}.xml")
    # print(midi_path)
    if os.path.exists(path):
        return 0
    if not os.path.exists(path):
        new_command = " ".join([command_prefix, midi_path, path,
                                "./test_def.xml"])
        os.system(new_command)
    return 0

np.random.seed(42)
random.seed(42)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    data_path = "../data/Piano"
    midi_sampled = os.listdir(data_path + f"/midi")
    os.makedirs(data_path +f"/feature", exist_ok=True)
    with Pool(processes=8) as pool:
        result = iter(tqdm(pool.imap(partial(get_jSymbolic_feature, root = data_path), midi_sampled),
                           total=len(midi_sampled)))
        for i in range(len(midi_sampled)):
            # try:
            next(result)







