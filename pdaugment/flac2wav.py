import os
import sys

try:
    flac_dir = sys.argv[1]
    output_wav_dir = sys.argv[2]
except IndexError:
    print("Need two command line parameters.")

list = os.listdir(flac_dir)

for sub_dir in list:
    if sub_dir == ".DS_Store":
        continue
    for s_sub_dir in os.listdir(os.path.join(flac_dir, sub_dir)):
        if s_sub_dir == ".DS_Store":
            continue
        output_dir = os.path.join(output_wav_dir, flac_dir.split("/")[-1], sub_dir, s_sub_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file in os.listdir(os.path.join(flac_dir, sub_dir, s_sub_dir)):
            if "trans.txt" in file:
                os.system("cp " + os.path.join(flac_dir, sub_dir, s_sub_dir, file) + " " + os.path.join(output_dir, file))
            else:
                os.system("ffmpeg -i " + os.path.join(flac_dir, sub_dir, s_sub_dir, file) + " " + os.path.join(output_dir, file.split(".")[0] + ".wav"))