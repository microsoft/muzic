#!/usr/bin/env python
# coding: utf-8

import io, os
import numpy as np


data_dir = 'l2m_merge'
output_dir = 'l2m_merge'
output_name = 'align_result.txt'

SEP   = '[sep]'
ALIGN = '[align]'
REST  = '128'

target_layer = 5

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

id2pitch = {0:'1',1:'#1', 2:'2', 3:'#2', 4:'3', 5:'4', 6:'#4', 7:'5', 8:'#5', 9:'6', 10:'#6', 11:'7'}
for i in range(12, 128):
    id2pitch[i] = id2pitch[i % 12]
id2pitch[128] = '-'


len_pen = 0.5
def dp(scores):
    mld_len = scores.shape[0]
    lrc_len = scores.shape[1]
    f = np.zeros(( mld_len+1, lrc_len+1))
    path = np.zeros((mld_len+1, lrc_len+1)).astype(np.int32) 
    # +x : (i-x, j-1)  -x: (i-1, j-1)
    f[:, :] = -np.inf
    f[0, 0] = 0
    
    scores = np.pad(scores,((1,0),(1,0)), 'constant')

    # 简单写不优化了
    for i in range(1,mld_len+1):
        for j in range(1, lrc_len+1):
            for k in range(i):
                scr = f[k, j-1] + np.mean(scores[k+1:i+1, j]) - len_pen*abs(i-k)/mld_len
                if scr > f[i,j]:
                    f[i,j] = scr
                    path[i, j] = i-k

            for k in range(j):
                scr = f[i-1, k] + np.mean(scores[i, k+1:j+1]) - len_pen*abs(j-k)/lrc_len # sum ？
                if scr > f[i, j]:
                    f[i,j] = scr
                    path[i, j] = -(j-k) 
     
    return path[1:, 1:]

def greedy(attn, melody_sent, lyric_sent):

    aligns_m = []
    aligns_l = []

    x = 0
    y = 0
    align_m = []
    align_l = []

    appx = True
    appy = True

    while (x<len(melody_sent)-1)and(y<len(lyric_sent)-1):
        if appx:
            align_m.append(melody_sent[x])
        if appy:
            align_l.append(lyric_sent[y])

        max_score = -1
        choice = 0
        if len(align_m) <= 1:
            # 多个lyric对一个m
            if attn[x, y+1] > max_score:
                max_score = attn[x,y+1]
                choice = 1

        if len(align_l) <= 1:
            # 多个melody对一个l：
            if attn[x+1, y] > max_score:
                max_score = attn[x+1,y]
                choice = 2

        if attn[x+1,y+1] > max_score:
            choice = 3

        if choice == 1:
            appx = False
            appy = True
            y += 1
            if y==len(lyric_sent)-1:
                aligns_m.append(align_m)
                aligns_l.append(align_l)
                x += 1
                break
            
        elif choice == 2:
            appx = True
            appy = False
            x += 1
            if x==len(melody_sent)-1:
                aligns_m.append(align_m)
                aligns_l.append(align_l)
                y += 1
                break
        else:
            aligns_m.append(align_m)
            aligns_l.append(align_l)
            align_m = []
            align_l = []
            appx=True
            appy=True
            x += 1
            y += 1

    aligns_m.append(melody_sent[x:])
    aligns_l.append(lyric_sent[y:])

    return aligns_m, aligns_l

def align_sent(melody_sent, lyric_sent, attn_sent):
    # melody pitch 和 duration 对应同一个
    # print(attn_sent.shape)
    pitch_attn_sent = attn_sent[np.array(range(0, attn_sent.shape[0], 2)), :]
    duration_attn_sent = attn_sent[np.array(range(1, attn_sent.shape[0], 2)), :]
    attn_sent = pitch_attn_sent + duration_attn_sent
    melody_sent = [(melody_sent[2*i], melody_sent[2*i+1]) for i in range(len(melody_sent)//2)] 

    # 休止符不参与解码
    not_rest_pitch_index = [i for i,x in enumerate(melody_sent) if x[0]!=REST ]
    melody_sent = [x for x in melody_sent if x[0] != REST]
    if len(melody_sent)==0:
        align_melody_sent = [ melody_sent ] if len(melody_sent) != 0 else [ [('128','129')] ]
        align_lyric_sent = [ lyric_sent ] if len(lyric_sent) != 0 else [ ['-'] ]
        return align_melody_sent, align_lyric_sent
    attn_sent = attn_sent[np.array(not_rest_pitch_index),:]

    # Dynamic Programming
    path = dp(attn_sent)

    align_melody_sent = []
    align_lyric_sent = []
    # Resolve Path
    mld = len(melody_sent)-1
    lrc = len(lyric_sent)-1
    while (mld >= 0 and lrc>= 0):
        offset = path[mld, lrc]
        if offset > 0:
            mld_last = mld - offset
            lrc_last = lrc - 1
        else:
            offset = -offset
            mld_last = mld - 1
            lrc_last = lrc - offset

        align_melody_sent.insert(0, melody_sent[mld_last+1:mld+1])
        align_lyric_sent.insert(0, lyric_sent[lrc_last+1:lrc+1])
        mld = mld_last
        lrc = lrc_last

    return align_melody_sent, align_lyric_sent

def align_song(idx):
    sample_dir = os.path.join(data_dir, idx)
    input_lyric_file = os.path.join(sample_dir, 'lyric')
    output_melody_file = os.path.join(sample_dir, 'melody')
    attn_file = os.path.join(sample_dir,'attn')


    with io.open(input_lyric_file,'r') as f:
        input_lyric = f.read().rstrip('\n').split(' ')
    with io.open(output_melody_file,'r') as f:
        output_melody = f.read().rstrip('\n').split(' ')
    with io.open(attn_file, 'r') as f:
        attn = np.array(eval(f.read().rstrip('\n'))) #num_layers, mld_len, lrc_len

    input_lyric = [x for x in input_lyric if x != ALIGN]

    lyric_sep = [i for i, x in enumerate(input_lyric) if x == SEP]
    lyric_sep.insert(0, -1)
    #已保证以SEP结尾
    melody_sep = [i for i,x in enumerate(output_melody) if x == SEP]
    melody_sep.insert(0,-1)

    aligned_lyric = []
    aligned_melody = []

    for idx in range(min(len(lyric_sep),len(melody_sep)) -1):
        lyric_start = lyric_sep[idx]+1
        lyric_end = lyric_sep[idx+1]
        melody_start =  melody_sep[idx]+1
        melody_end = melody_sep[idx+1]


        melody_sent = output_melody[melody_start:melody_end]
        lyric_sent = input_lyric[lyric_start:lyric_end]

        attn_sent = attn[target_layer, melody_start:melody_end, lyric_start:lyric_end]

        if len(melody_sent) != 0 and len(lyric_sent) != 0:
            align_melody_sent, align_lyric_sent = align_sent(melody_sent, lyric_sent, attn_sent)
        else:
            melody_sent = [(melody_sent[2*i], melody_sent[2*i+1]) for i in range(len(melody_sent)//2)]
            align_melody_sent = [ melody_sent ] if len(melody_sent) != 0 else [ [('128','129')] ]
            align_lyric_sent = [ lyric_sent ] if len(lyric_sent) != 0 else [ ['-'] ]

        aligned_lyric.extend(align_lyric_sent)
        aligned_melody.extend(align_melody_sent)
        aligned_lyric.append(SEP)
        aligned_melody.append(SEP)

    return aligned_melody, aligned_lyric

def pad_length(melody_str, lyric_str):
    if len(melody_str) < len(lyric_str):
        melody_str = melody_str + "".join( [" "]*(len(lyric_str)-len(melody_str)) )
    elif len(lyric_str) < len(melody_str):
        lyric_str = lyric_str + "".join( [" "]*(len(melody_str)-len(lyric_str)) )
    return melody_str, lyric_str

def write_align_part(melody, lyric):
    if len(melody) == 1:
        melody_str = id2pitch[int(melody[0][0])]
        lyric_str = lyric[0]

        melody_str, lyric_str = pad_length(melody_str, lyric_str)
        for i in range(1, len(lyric)):
            melody_str = melody_str + " " + "-"
            lyric_str = lyric_str + " " + lyric[i]
            melody_str, lyric_str = pad_length(melody_str, lyric_str)
        return melody_str, lyric_str
    else:
        melody_str = " ".join( [ id2pitch[int(x[0])] for x in melody ])
        lyric_str = lyric[0]
        melody_str, lyric_str = pad_length(melody_str, lyric_str)
        return melody_str, lyric_str

def write_song(aligned_melody, aligned_lyric, f=None):
    melody_str_list = []
    lyric_str_list = []
    for i in range(len(aligned_lyric)):
        if aligned_lyric[i] == SEP:
            if f is None:
                print(" ".join(melody_str_list))
                print(" ".join(lyric_str_list))
            else:
                f.write(" ".join(melody_str_list) + '\n')
                f.write(" ".join(lyric_str_list) + '\n')
            melody_str_list = []
            lyric_str_list = []     
            continue
            
        align_part_melody_str, align_part_lyric_str = write_align_part(aligned_melody[i], aligned_lyric[i])
        melody_str_list.append(align_part_melody_str)
        lyric_str_list.append(align_part_lyric_str)

if __name__ == "__main__":
    for d in os.listdir(data_dir):
        aligned_melody, aligned_lyric = align_song(d)
        with io.open(os.path.join(output_dir, f"{d}/{output_name}"), 'w') as f:
            write_song(aligned_melody, aligned_lyric, f)