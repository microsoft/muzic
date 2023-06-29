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
from reference.MidiProcessor.midiprocessor import MidiEncoder, midi_utils
from tqdm import tqdm
import gc
from multiprocessing import Pool, Manager, Lock
import math, random
from typing import List, Dict



# INSTR_NAME_LIST = [0, 1, 4, 5, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35, 48, 49, 50, 52, 53, 56, 57, 60, 61, 65, 66, 71, 73, 128]
INSTR_NAME_LIST = [0, 25, 32, 48, 80, 128]
# INSTR_NAME_LIST = [0, 25, 32, 48, 128]
#
#
#
# # Banlanced_INST = [1, 4, 5, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35, 48, 49, 50, 52, 53, 56, 57, 60, 61, 65, 66, 71, 73]
INSTR_NAME2_index = dict(zip(INSTR_NAME_LIST, range(len(INSTR_NAME_LIST))))
sample_INSTR_dist = dict(zip(INSTR_NAME_LIST, [0]*len(INSTR_NAME_LIST)))
file_path = './test.mid'
# sample_list = dict.fromkeys(INSTR_NAME_LIST, 0)


encoder = MidiEncoder('REMIGEN2')

ignore_inst = False

def seed_everything():
    np.random.seed(42)
    random.seed(42)


# def gen_encode_data(): # generate command_vector from txt file,这个不用了，产生小样本用的
#     data_root = "./data/lmd_encode"
#     start_token, end_token = "<s>", "</s>"
#     dict = []
#
#     # with open("./data/txt/dict.txt", "a") as f:
#     #     for i in range(129):
#     #         f.write(f"i-{i} 1\n")
#     #   bos = "<s>",
#     #         pad="<pad>",
#     #         eos="</s>",
#     # dict = []  # use dict from longformer_exps
#     for target_prefix in ["train", "valid", "test"]:
#         file_count = 0
#         target_file = open(f"./data/txt/{target_prefix}.txt", "w")
#         # cf = open(f"./data/txt/command_{target_prefix}.txt", "w")
#         cf = []
#         index_f = open(f"./data/txt/index_{target_prefix}.id", "w")
#         for file in os.listdir(data_root):
#             # print(target_prefix, file_count)
#             with open(os.path.join(data_root, file), "r") as f:
#                 remi_str = f.read().strip("\n")
#                 target_file.write(" ".join([start_token, remi_str, end_token]) + "\n")
#                 command_vector = [0]*len(INSTR_NAME_LIST)
#                 for i in remi_str.split(" "):
#                     if i not in dict:
#                         dict.append(i)
#                     if i[0] == "i":
#                         instru_name_now = eval(i[2:])
#                         assert instru_name_now in INSTR_NAME_LIST, f"{file} illegal!"
#                         command_vector[INSTR_NAME2_index[instru_name_now]] = 1
#
#                 cf.append(command_vector)
#                 # cf.write(" ".join(command_vector) + "\n")
#                 index_f.write(file.split(".")[0] + "\n")
#             file_count += 1
#             if file_count >= 117:
#                 break
#         cf = np.array(cf)
#         np.save(f"./data/data_bin/command_{target_prefix}.npy", cf)
#     target_file.close()
#     # dict = sorted(dict, key = lambda x: (x[0], eval(x.split("-")[-1])))
#     # with open("./data/txt/dict.txt", "w") as f:
#     #     for i in dict:
#     #         f.write(i + " 1\n")
#     write_dict(dict, "./data/txt/dict.txt")
#     print("done!")

def check_ts(midi_obj):
    for i in midi_obj.time_signature_changes:
        a, b = i.numerator, i.denominator
        if not(b == 4 and a in [4,8,16,32,64,128,256]):
            return False
    return True

# def ergodic_a_file(path_root:str, path_list: List[str], dict_list:List[str], locker, process_index: int, processor_num: int, finished_num):
#
#     # for i in range(1000):
#     #     locker.acquire()
#     #     finished_num.value += 1
#     #     locker.release()
#     #     print(process_index, finished_num.value)
#
#     size = math.ceil(len(path_list) / processor_num)
#     start = size * process_index
#     end = (process_index + 1) * size if (process_index + 1) * size < len(path_list) else len(path_list)
#     for path in path_list[start:end]:
#         with open(os.path.join(path_root, path), "r") as f:
#             line = f.read().strip("\n").split(" ")
#             for token in line:
#                 if token not in dict_list:
#                     locker.acquire()
#                     dict_list.append(token)
#                     locker.release()
#
#         # if process_index == 2:
#         locker.acquire()
#         finished_num.value += 1
#         locker.release()
#         print(process_index, finished_num.value)
#         # if finished_num.value > 100:
#         #     break
#
# def generate_dict(dataset_path:str, save_root:str):
#     # dataset_path: 输入的remi编码的根目录, 文件夹里是remi的txt序列
#     # save_root: 存储的dict 根目录
#
#     file_name_list = os.listdir(dataset_path)
#     # dict_list = []
#     manager = Manager()
#     dict_list = manager.list()
#     locker = manager.Lock()
#     finished_num = manager.Value("i", 0)
#     processor_num = 4
#
#
#     process_list = []
#     for i in range(processor_num):
#         # res.append(pools.apply_async(pass_para, args=(file_name_list, i, processor_num, sample_list, finished_num,)))
#         process_list.append(multiprocessing.Process(target=ergodic_a_file, args=(
#             dataset_path, file_name_list, dict_list, locker, i, processor_num, finished_num,)))
#         process_list[-1].start()
#         print(str(i) + ' processor started !')
#
#     for i in process_list:
#         i.join()
#     print("over!")
#
#     # for file_name in tqdm(file_name_list):
#     #     file_path = os.path.join(dataset_path, file_name)
#     dict_list = sorted(dict_list, key=lambda x: (x[0], eval(x.split("-")[-1])))
#     with open(os.path.join(save_root, "dict.txt"), "w") as f:
#         for i in dict_list:
#             f.write(f"{i} 1\n")
#     return dict_list

def midi2REMI(midi_obj, save_path, sample_list = None):
    if len(midi_obj.instruments) > 0:
        encoder.encode_file(
            file_path,
            midi_obj=midi_obj,
            remove_empty_bars=True,
            ignore_inst=ignore_inst,
            ignore_ts=True,
            ignore_tempo=False,
            save_path = save_path
        )

def pass_para(file_in, index, p_num, sample_list, finished_num):
    # if eval(file_in[1]):
    #     return
    # file_name = file_in[0]
    # save_name = "./data/lmd_encode/" + file_name.replace("/", "_")[:-4] + ".txt"
    # midi_obj = miditoolkit.MidiFile("./data/lmd_full/" + file_name)
    # midi2REMI(midi_obj, save_name)
    size = math.ceil(len(file_in) / p_num)
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(file_in) else len(file_in)
    temp_data = file_in[start:end]
    for j in temp_data:
        # if eval(j[1]):
        #     continue
        file_name = j
        # save_name = r"E:\AIMusic\ControlGeneration\lmd_6_tracks_encode/remi_seq/" + file_name.replace("/", "_")[:-4] + ".txt"
        save_name = r"D:\ProjectData\ControlGeneration\0813_attributes_generation\lmd_6tracks_clean\full_song\remi_seq/" + file_name.split(".")[0] + ".txt"
        # if os.path.exists(save_name):
        #     if index == 2:
        #         print("skip")
        #     continue
        midi_obj = miditoolkit.MidiFile(os.path.join(r"D:\ProjectData\datasets\lmd_6tracks\midi_6tracks", file_name))
        if check_ts(midi_obj):
            midi_obj.time_signature_changes = [miditoolkit.TimeSignature(4,4,0)]
            try:
                midi2REMI(midi_obj, save_name, sample_list)
            except:
                print(file_name, " illegal!!!")
        finished_num.value += 1
        if index == 2:
            print(finished_num.value)

def mutiprocess_REMI_encode(file_name_list):
    # ex_item = pd.read_excel("./data/LMD_full_statistics.xlsx", header = [0,1])
    # file_name_list = np.load("./data/file_path.npy")
    # for i in tqdm(range(50000)):
    #     if not eval(file_name_list[i,1]):
    #         save_name = "./data/lmd_encode/" + file_name_list[i,0].replace("/", "_")[:-4] + ".txt"
    #         midi_obj = miditoolkit.MidiFile("./data/lmd_full/" + file_name_list[i,0])
    #         midi2REMI(midi_obj, save_name)
    #         del midi_obj
    #         gc.collect()

    manager = Manager()
    sample_list = manager.dict()
    finished_num = manager.Value("i", 0)

    processor_num = 4
    process_list = []
    for i in range(processor_num):
        # res.append(pools.apply_async(pass_para, args=(file_name_list, i, processor_num, sample_list, finished_num,)))
        process_list.append(multiprocessing.Process(target=pass_para, args=(
        file_name_list, i, processor_num, sample_list, finished_num,)))
        process_list[-1].start()
        print(str(i) + ' processor started !')

    for i in process_list:
        i.join()
    print("over!")
    # save_dict = {}
    # for i in sample_list.items():
    #     print("key:", type(i[0]))
    #     save_dict[int(i[0])] = i[1]
    # json.dump(save_dict, open("./data/sample_list.json", "w"))

# def sample_midi_data():
#     # sample_list = json.load(open("./data/sample_list.json", "r"))
#     file_name_list = np.load("./data/file_instru.npy", allow_pickle=True)
#
#     chosen_file = []
#     total_dist = dict(zip(range(129), [0] * 129))
#     for file in tqdm(file_name_list):
#         if file[1]:
#             continue
#         # midi_obj = miditoolkit.MidiFile("./data/lmd_full/" + file[0])
#         isSelected = False
#         for instru in eval(file[2]):
#             if instru in Banlanced_INST:
#                 if not isSelected:
#                     chosen_file.append(file)
#                     isSelected = True
#
#             total_dist[min(instru, 128)] += 1
#         if isSelected:
#             for instru in eval(file[2]):
#                 if instru in INSTR_NAME_LIST or instru >= 128:
#                     sample_INSTR_dist[min(instru, 128)] += 1
#     np.save("./data/chosen_files.npy", chosen_file)
#     np.save("./data/selected_dist.npy", sample_INSTR_dist)
#     # {0: 46250, 1: 15661, 4: 13041, 5: 10634, 24: 23789, 25: 30324, 26: 16262, 27: 21368, 28: 13430, 29: 16726, 30: 16890, 32: 22399, 33: 36293,
#     # 35: 21158, 48: 37318, 49: 20172, 50: 12922, 52: 20990, 53: 10138, 56: 17569, 57: 13767, 60: 11161, 61: 13343, 65: 14791, 66: 11622, 71: 10262, 73: 16688, 128: 97225}

# def gen_train_valid_test_txt():
#     np.random.seed(42)
#     # chosen_file = np.load("./data/chosen_files.npy", allow_pickle=True)
#     root = "E:\AIMusic\ControlGeneration\lmd_6_tracks_encode\clean"
#     chosen_file = os.listdir(root)
#     split_point = [int(0.88*len(chosen_file)), int(0.95*len(chosen_file))]
#     chosen_index = np.arange(0,len(chosen_file))
#     np.random.shuffle(chosen_index)
#     train_file_index = chosen_index[:split_point[0]]
#     valid_file_index = chosen_index[split_point[0]:split_point[1]]
#     test_file_index = chosen_index[split_point[1]:]
#     # dict = []
#     # np.save("./data/total_txt/split_index.npy", [train_file_index, valid_file_index, test_file_index])
#     def get_split_txt(split, indexes):
#         # command_list = []
#         f = open(rf"E:\AIMusic\ControlGeneration\instrument_control_bar\attribute_control\txt/{split}.txt", "w")
#
#         num = 0
#         print(split, " start!")
#         for i in tqdm(indexes):
#             file_path = os.path.join(root, chosen_file[i])
#             s_encode = open(file_path, "r").read()
#             if s_encode[-1:] != "\n":
#                 s_encode = s_encode + "\n"
#             f.write(s_encode)
#             num += 1
#             # if num > 100:
#             #     break
#         f.close()
#         # np.save(f"./data/total_txt/command_{split}.npy", np.array(command_list))
#     get_split_txt("train", train_file_index)
#     get_split_txt("valid", valid_file_index)
#     get_split_txt("test", test_file_index)
#     # write_dict(dict, "./data/total_txt/dict.txt")
#     print("done!")
#
# def check(check_index):
#     command = np.load("./data/total_txt/command_train.npy")
#     index = 0
#     with open("./data/total_txt/train.txt", "r") as f:
#         line = f.readline().strip("\n")
#         while 1:
#             # for i in line.split(" "):
#             #     if i[0] == "i":
#             #         instru_number = eval(i[2:])
#             #         assert command[index][INSTR_NAME2_index[instru_number]] == 1, "error!"
#
#             if check_index == index:
#                 print(line)
#                 break
#             index += 1
#             line = f.readline().strip("\n")
#             if index % 1000 == 0:
#                 print(index)
#
# # def generate_command_vector(token_list:list):
# #     command_vector = [0] * (len(INSTR_NAME_LIST))
# #     length = 0
# #     for i in token_list:
# #         if i == "":
# #             continue
# #         if i[0] == "i":
# #             j = eval(i[2:])
# #             command_vector[INSTR_NAME2_index[min(j, 128)]] = 1
# #         length += 1
# #     # command_vector = np.array([command_vector])
# #     # command_vector = np.repeat(command_vector, length, axis = 0)
# #     command_vector.append(length + 1) # for b-1 token
# #     return command_vector
#
# def get_8bars_train_valid_test_txt():
#     # 随机截取8个bar长度的token试一试
#     # chosen_file = np.load("./data/chosen_files.npy", allow_pickle=True)
#     # data_path = "E:/AIMusic/ControlGeneration/lmd_encode/"
#     # save_root = "E:/AIMusic/ControlGeneration/8-bars-txt/"
#     # data_path = "./data/sample_6tracks/remi_seq/"
#     # save_root = "./data/sample_6tracks/txt/"
#     data_path = "E:/AIMusic/ControlGeneration/lmd_6_tracks_encode/remi_seq/"
#     save_root = "E:/AIMusic/ControlGeneration/lmd_6_tracks_encode/txt/"
#
#     chosen_file = [i for i in os.listdir(data_path)]
#
#     def write(code, target_f):
#         for i in range(len(code)):
#             if code[i] == "s-9":
#                 continue
#             target_f.write(code[i])
#             if i != len(code)-1:
#                 target_f.write(" ")
#         target_f.write("\n")
#     def get_bar_split_txt(split, start, end):
#         command_list = []
#         split_files = chosen_file[start:end]
#         f = open(save_root + f"./{split}.txt", "w")
#         num = 0
#         print(split, " start!")
#         max_length = 3000
#         sample_num = 0
#         for file_name in tqdm(split_files):
#             file_path = data_path + file_name.split(".")[0].replace("/", "_") + ".txt"
#             s_encode = open(file_path, "r").read().strip("\n")
#             if s_encode[:3] != "s-9":
#                 continue
#             bar_pos = []
#             remi_tokens = s_encode.split(" ")
#             for i, token in enumerate(remi_tokens):
#                 if token == "b-1":
#                     bar_pos.append(i)
#
#             if len(bar_pos) <= 8:
#                 cur_length = len(remi_tokens) -1
#                 if cur_length + 2 <= max_length:
#                     write(remi_tokens[0:], f)
#                     sample_num += 1
#                 continue
#
#             coordinate_pool = []
#             start = 0
#             for i, pos in enumerate(bar_pos[:-7]):
#                 cur_length = bar_pos[i+7] + 1 - start
#                 if cur_length -8 + 2 <= max_length:
#                     coordinate_pool.append((i, start, start + cur_length))
#                 start = pos + 1
#             if len(coordinate_pool) == 0:
#                 continue
#             elif len(coordinate_pool) == 1:
#                 arr = coordinate_pool[0]
#                 write(remi_tokens[arr[1]:arr[2]], f)
#                 sample_num += 1
#             else:
#                 chosen_bars = np.random.choice(len(coordinate_pool), 2, replace = False)
#                 for i in chosen_bars:
#                     arr = coordinate_pool[i]
#                     write(remi_tokens[arr[1]:arr[2]], f)
#                     sample_num += 1
#         print(sample_num)
#         f.close()
#         # np.save(f"./data/total_txt/command_{split}.npy", np.array(command_list))
#     get_bar_split_txt("train", 0, 70000)
#     get_bar_split_txt("valid", 70000, 75000)
#     get_bar_split_txt("test", 75000, 80000)
# def statics_length():
#     # 统计token长度
#     f = open("./data/total_txt/train.txt", "r")
#     line = f.readline()
#     length_dict = {}
#     num = 0
#     while line:
#         length = len(line.split(" "))
#         if length not in length_dict:
#             length_dict[length] = 1
#         else:
#             length_dict[length] += 1
#         line = f.readline()
#         num += 1
#         print("\r", num, end = "")
#
#     import matplotlib.pyplot as plt
#
#     X = length_dict.keys()
#     Y = length_dict.values()
#     fig = plt.figure()
#     plt.bar(X, Y, 0.4, color="green")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
#     plt.title("bar chart")
#     plt.show()
#     ave = 0
#     for j in length_dict.items():
#         ave += j[0]*j[1]/num
#     ave = 14995.75
#
# def dict_generation(path):
#     dict = []
#     for file in os.listdir(path):
#         f = open(os.path.join(path, file))
#         line = f.readline()
#         while line:
#             tokens = line.strip("\n").split(" ")
#             assert len(tokens) <= 3010, "too long token sequence"
#             for j in tokens:
#                 if j not in dict:
#                     dict.append(j)
#             line = f.readline()
#         f.close()
#     g = open(os.path.join(path, "dict.txt"), "w")
#     dict = sorted(dict, key=lambda x: (x[0], eval(x.split("-")[-1])))
#     for i in dict:
#         g.write(i + " 1\n")
#     g.close()
#
# def lmd_6tracks_clean():
#     # 清理lmd_6tracks的代码
#     chosen_files = np.load(r"./data/lmd_full_not_duplicate_files.npy", allow_pickle=True)
#     all_files = np.load(r"E:\AIMusic\ControlGeneration\全局的控制数据-28类乐器\file_path.npy", allow_pickle=True)
#     all_file_names = []
#     for i in tqdm(range(len(all_files))):
#         cur_name = all_files[i][0]
#         all_file_names.append(cur_name)
#
#     out_of_lmd_full_num = 0
#     successs_num = 0
#     error_name_list = []
#     for tracks6_name in tqdm(os.listdir(r"E:\AIMusic\ControlGeneration\lmd_6_tracks_encode\remi_seq")):
#         search_name = tracks6_name.split("_")
#
#         search_name = search_name[-1][0] + "/" + search_name[-1].split(".")[0] + ".mid"
#         if len(search_name) != 38:
#             error_name_list.append(tracks6_name)
#         if search_name not in all_file_names:
#             out_of_lmd_full_num += 1
#         else:
#             successs_num += 1
#             source_path = os.path.join(r"E:\AIMusic\ControlGeneration\lmd_6_tracks_encode\remi_seq", tracks6_name)
#             with open(source_path, "r") as f:
#                 line = f.readline().strip("\n")
#                 line = line.split(" ")
#                 is4_4 = True
#                 for j in line:
#                     if j[0] == "s" and eval(j[2:]) != 9:
#                         is4_4 = False
#                         break
#                 if is4_4:
#                     target_path = os.path.join(r"E:\AIMusic\ControlGeneration\lmd_6_tracks_encode\clean", tracks6_name)
#                     shutil.copy(source_path, target_path)
#     print(out_of_lmd_full_num, successs_num)
#
#     length = []
#     for i in all_file_names:
#         if len(i) not in length:
#             length.append((len(i)))
#
#     length_list = {}
#     length_33 = []
#     for name in error_name_list:
#         if len(name)-4 not in length_list.keys():
#             length_list[len(name)-4] = 1
#         else:
#             length_list[len(name)- 4] += 1
#         if len(name)-4 == 33:
#             length_33.append(name)

if __name__ == "__main__":

    # dict = generate_dict(r"E:\AIMusic\ControlGeneration\lmd_6_tracks_encode\clean_4_4", r"E:\AIMusic\ControlGeneration\lmd_6_tracks_encode")
    # x = 1
    # name_list = os.listdir("./data/lmd_6tracks/midi_6tracks/")
    #
    # name_list = np.random.choice(name_list, 10000, replace=False)
    # np.save("./data/chosen_files_6tracks.npy", name_list)

    # file_name_list = np.load("data/not_duplicate_list.npy")
    # file_name_list = os.listdir(r"D:\ProjectData\datasets\lmd_6tracks\midi_6tracks")
    # print(len(file_name_list))
    file_name_list = np.load(r"D:\ProjectData\datasets\lmd_6tracks_clean_4_4\file_name_list.npy")
    mutiprocess_REMI_encode(file_name_list)

    # gen_train_valid_test_txt()

    # file_name_list = os.listdir("E:\AIMusic\ControlGeneration\lmd_6_tracks_encode\clean_4_4")


    # test_mid_obj = miditoolkit.MidiFile("./test.mid")
    # midi2REMI(test_mid_obj, )
    # file_name_list = np.load("./data/file_path.npy")
    # for file_name in tqdm(file_name_list[:,0]):
    #     save_name = r"E:\AIMusic\ControlGeneration\lmd_encode/" + file_name.replace("/", "_")[:-4] + ".txt"
    #     if os.path.exists(save_name):
    #         continue
    #     midi_obj = miditoolkit.MidiFile("./data/lmd_full/" + file_name)
    #     midi2REMI(midi_obj, save_name, {})

    # get_8bars_train_valid_test_txt()
    # dict_generation(r"E:\AIMusic\ControlGeneration\lmd_6_tracks_encode\txt/")
    # command_generation(r"E:\AIMusic\ControlGeneration\lmd_6_tracks_encode\txt/")


    # path = "./data/sample_6tracks/midi"
    # save_root = "./data/sample_6tracks/remi_seq/"
    # for i, file in enumerate(os.listdir(path)):
    #     midi_obj = miditoolkit.MidiFile(os.path.join(path, file))
    #     midi2REMI(midi_obj, os.path.join(save_root, file.split(".")[0] +".txt"))



    # fairseq bug:
    # g = open("./data/total_txt/train.txt", "w")
    # with open("./data/total_txt/train_total.txt", "r") as f:
    #     line = f.readline()
    #     index = 0
    #     while 1:
    #         # for i in line.split(" "):
    #         #     if i[0] == "i":
    #         #         instru_number = eval(i[2:])
    #         #         assert command[index][INSTR_NAME2_index[instru_number]] == 1, "error!"
    #         g.write(line)
    #         index += 1
    #         line = f.readline()
    #         if index % 1000 == 0:
    #             print(index)
    #         if index > 70000:
    #             break
    # gen_train_valid_test_txt()
    # check(90000)
    # def _warmup_mmap_file(path):
    #     with open(path, "rb") as stream:
    #         while stream.read(100 * 1024 * 1024):
    #             pass
    # import struct
    # dtypes = {
    #     1: np.uint8,
    #     2: np.int8,
    #     3: np.int16,
    #     4: np.int32,
    #     5: np.int64,
    #     6: np.float,
    #     7: np.double,
    #     8: np.uint16,
    # }
    # path ="./data/total-data-bin/train.idx"
    # _HDR_MAGIC = b"MMIDIDX\x00\x00"
    # with open(path, "rb") as stream:
    #     magic_test = stream.read(9)
    #     assert _HDR_MAGIC == magic_test, (
    #         "Index file doesn't match expected format. "
    #         "Make sure that --dataset-impl is configured properly."
    #     )
    #     version = struct.unpack("<Q", stream.read(8))
    #     assert (1,) == version
    #
    #     (dtype_code,) = struct.unpack("<B", stream.read(1))
    #     _dtype = dtypes[dtype_code]
    #     _dtype_size = _dtype().itemsize
    #
    #     _len = struct.unpack("<Q", stream.read(8))[0]
    #     offset = stream.tell()
    #
    # _warmup_mmap_file(path)
    #
    # _bin_buffer_mmap = np.memmap(path, mode="r", order="C")
    # _bin_buffer = memoryview(_bin_buffer_mmap)
    # _sizes = np.frombuffer(
    #     _bin_buffer, dtype=np.int32, count=_len, offset=offset
    # )
    # _pointers = np.frombuffer(
    #     _bin_buffer,
    #     dtype=np.int64,
    #     count=_len,
    #     offset=offset + _sizes.nbytes,
    # )

    # fairseq-preprocess --only-source --srcdict data/total_txt/dict.txt --trainpref data/total_txt/train.txt --validpref data/total_txt/valid.txt --testpref data/total_txt/test.txt --destdir data/total-data-bin --workers 8

    # midi_root_path = r"D:\ProjectData\datasets\lmd_full"
    # save_root = r"D:\Project\ControlGeneration\StyleCtrl\jSymbolic_lib\datasets\TopMAGD/midi"
    # midi_to_genre = json.load(open("./midi_genre_map.json", "r"))
    # top_magd_genre = midi_to_genre["topmagd"]
    # for key in tqdm(top_magd_genre.keys()):
    #     midi_name = key + ".mid"
    #     midi_src_path = os.path.join(midi_root_path, midi_name[0], midi_name)
    #     midi_save_path = os.path.join(save_root, midi_name)
    #     shutil.copyfile(midi_src_path, midi_save_path)

    # midi_root_path = r"D:\ProjectData\datasets\lmd_full"
    # save_root = r"D:\Project\ControlGeneration\StyleCtrl\jSymbolic_lib\datasets\MASD/midi"
    # midi_to_genre = json.load(open("./midi_genre_map.json", "r"))
    # masd_genre = midi_to_genre["masd"]
    # for key in tqdm(masd_genre.keys()):
    #     midi_name = key + ".mid"
    #     midi_src_path = os.path.join(midi_root_path, midi_name[0], midi_name)
    #     midi_save_path = os.path.join(save_root, midi_name)
    #     shutil.copyfile(midi_src_path, midi_save_path)
