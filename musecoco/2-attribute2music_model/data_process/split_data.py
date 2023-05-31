import os
import random
import numpy as np
import pandas as pd
import json
from util import seed_everything
import msgpack
seed_everything(2023)
from tqdm import tqdm


def split_data(path, save_path, truncated_length = 5120):
    os.makedirs(save_path, exist_ok=True)
    max_idx = 1000
    with open(path + '/RID.bin', 'rb') as RID:
        with open(path + f"/TOKEN.bin", "rb") as TOKEN:
            total_index = list(range(max_idx))
            random.shuffle(total_index)
            step = [0, 0.95, 0.975, 1.0]
            split_txt_pool = {}
            split_command_pool = {}
            target_split = {}
            for idx, split in enumerate(["train", "valid", "test"]):
                s, e = step[idx:idx + 2]
                s, e = int(s * max_idx), int(e * max_idx)
                for i in total_index[s:e]:
                    target_split[i] = split
                split_txt_pool[split] = [open(save_path + f"/{split}.txt", "w"), 0]
                split_command_pool[split] = []

            RID_unpacker = msgpack.Unpacker(RID, use_list=False)
            TOKEN_unpacker = msgpack.Unpacker(TOKEN, use_list=False)
            for i in tqdm(range(max_idx)):
                rid_info = next(RID_unpacker)
                tokens = next(TOKEN_unpacker)
                split = target_split[i]
                for piece in rid_info["pieces"]:
                    token_s, token_e = piece["token_begin"], piece["token_end"]
                    if token_e - token_s > truncated_length: # skip when sequence > max_length
                        continue
                    split_txt_pool[split][0].write(" ".join(tokens[token_s:token_e]) + f'\n')
                    split_txt_pool[split][1] += 1
                    split_command_pool[split].append([piece["values"]])
                    if split_txt_pool[split][1] % 300000 == 0:
                        split_txt_pool[split][0].close()
                        split_txt_pool[split][0] = open(save_path + f"/{split}.txt", "a")
            for split in ["train", "valid", "test"]:
                np.save(save_path + f"/{split}_command.npy", split_command_pool[split])
                print("sample num:", split, len(split_command_pool[split]))
if __name__ == "__main__":
    path = "../data"
    truncated_length = 5120
    save_path = "data" + f"/truncated_{truncated_length}"
    split_data(path, save_path, truncated_length=truncated_length)



