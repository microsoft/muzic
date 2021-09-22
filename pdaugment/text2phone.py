from phonemizer import phonemize
from phonemizer.separator import Separator
import sys
import os
import csv

try:
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]
except IndexError:
    print("Need two command line parameters, e.g, python text2phone.py <dataset_dir> <output_dir>.")
    exit(0)

headers = ['wav', 'new_wav', 'txt', 'phone', 'new_phone']
rows = []

list = os.listdir(dataset_dir)

for sub_dir in list:
    for s_sub_dir in os.listdir(os.path.join(dataset_dir, sub_dir)):
        with open(os.path.join(dataset_dir, sub_dir, s_sub_dir, sub_dir + "-" + s_sub_dir + ".trans.txt"), "r") as tran:
            for line in tran.readlines():
                new_wav = line.replace("\n", "").split(" ")[0] + ".wav"
                txt = " ".join(line.replace("\n", "").split(" ")[1:]).lower()
                wav = os.path.join(dataset_dir, sub_dir, s_sub_dir, new_wav)
                phone = phonemize(txt, separator=Separator('/',phone=' ', syllable="-")).replace("-/", "/ ").replace("-", "- ") + "punc_."
                new_phone = "<BOS> " + phonemize(txt, separator=Separator(word=None, phone=' ', syllable=None)) + "<EOS>"
                rows.append([wav, new_wav, txt, phone, new_phone])
            tran.close()
        print(os.path.join(dataset_dir, sub_dir, s_sub_dir, sub_dir + "-" + s_sub_dir + ".trans.txt") + " done!")
with open(os.path.join(output_dir, dataset_dir.split("/")[-1] + "_metadata.csv"),'w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
    f.close()
