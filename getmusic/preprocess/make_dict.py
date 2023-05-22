from tqdm import tqdm
import sys
import os
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from getmusic.utils.midi_config import *
import os
import pickle
prog_to_name = {'32':'Bass','128':'Drums','25':'Guitar','0':'Grand_Piano','48':'Strings'}

dur_vocab_size = duration_max * pos_resolution
# 9 celesta
# 25 acoustic guitar(steel)
# 32 acoustic bass
# 48 string ensemble
# 80 synth
# 128 drum
root_dict = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
kind_dict = {'null': 0, 'm': 1, '+': 2, 'dim': 3, 'seven': 4, 'maj7': 5, 'm7': 6, 'm7b5': 7}

dpath = sys.argv[1]
output_txt_path = dpath
num_threshold = int(sys.argv[2])
oct_path = os.path.join(dpath,'oct.txt')

with open(oct_path,'r') as f:
    lines = f.readlines()
    token_dict = {'32':{},'128':{},'25':{},'0':{},'48':{}}
    for line in tqdm(lines):
        if '<2-80>' not in line:
            continue

        track_numbers = ('<2-0>' in line) + ('<2-25>' in line) + ('<2-32>' in line) + ('<2-48>' in line) + ('<2-128>' in line)
        
        if track_numbers == 0:
            continue

        tmp = line.strip().split('<0-')[1:]
        encoding = []
        for item in tmp:
            tmp2 = item.strip()[:-1].split('> <')
            encoding.append([int(tmp2[0])] + [int(i[2:]) for i in tmp2[1:]])
        
        idx = 0
        while idx != len(encoding) - 1:
            e = encoding[idx]

            if e[2] == 80 or e[2] == 129 or e[2] not in [0,25,32,48,80,128]:
                idx += 1
                continue

            bar = e[0]
            pos = e[1]
            inst = e[2]
            pitch = e[3]
            dur = e[4]

            chord_list = [str(e[3])]
            dur_list = [e[4]]

            for f_idx in range(idx + 1, len(encoding)):
                if (encoding[f_idx][0] == bar) and (encoding[f_idx][1] == pos) and (encoding[f_idx][2] == inst):
                    if encoding[f_idx][3] != pitch:
                        chord_list.append(str(encoding[f_idx][3]))
                        dur_list.append(encoding[f_idx][4])
                        pitch = encoding[f_idx][3]
                else:
                    break
            
            if len(chord_list) > 1:
                chord_str = ' '.join(chord_list)
                if chord_str in token_dict[str(inst)]:
                    token_dict[str(inst)][chord_str] += 1
                else:
                    token_dict[str(inst)][chord_str] = 1

            idx = max(idx + 1, f_idx)

output_txt_path = os.path.join(output_txt_path, 'pitch_dict.txt')
w = open(output_txt_path,'w')

for j in range(1, dur_vocab_size):
    w.write('T'+str(j)+'\t'+'0\n') # duration
w.write('<pad>\t0\n')

for j in range(0,128):
    w.write('M'+str(j)+'\t'+'0\n') # pitch

start_ls = [dur_vocab_size]
end_ls = [dur_vocab_size + 127]

start_idx = dur_vocab_size + 128
for idx, inst_program in enumerate(token_dict):
    start_ls.append(start_idx)
    threshold = sum(token_dict[inst_program].values()) * 0.9
    inst_list_of_tuples = [(k, v) for k, v in token_dict[inst_program].items()]

    inst_list_of_tuples = sorted(inst_list_of_tuples, key=lambda tup: tup[1], reverse=True)

    if inst_program == '128':
        for j in range(128,256):
            w.write('D'+str(j)+'\t'+'0\n') # pitch
    elif inst_program != '0':
        for j in range(0,128):
            w.write(prog_to_name[inst_program][0]+str(j)+'\t'+'0\n') # pitch
    else:
        for j in range(0,128):
            w.write('P'+str(j)+'\t'+'0\n') # pitch

    chord_sum = 0
    for idx,j in enumerate(inst_list_of_tuples):
        chord_sum += j[1]
        if chord_sum < threshold and j[1] > num_threshold:
            if inst_program != '0': 
                w.write(prog_to_name[inst_program][0] + j[0] + '\t' + str(j[1]) + '\n')
            else:
                w.write('P' + j[0] + '\t' + str(j[1]) + '\n')
        else:
            print(prog_to_name[inst_program])
            print("total tokens:",int(1.1 * threshold))
            print('we keep {}, which is {} ({}%) tokens'.format(chord_sum, idx, chord_sum / (1.1 * threshold)))
            break
    start_idx += 128 + idx - 1
    end_ls.append(start_idx)
    start_idx += 1

print(start_ls)
print(end_ls)
for r in root_dict:
    w.write(r+'\t'+'0\n')

for k in kind_dict:
    w.write(k+'\t'+'0\n')

w.close()