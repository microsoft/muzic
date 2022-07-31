#!/usr/bin/env python
# coding: utf-8
import io
import os
import shutil
import sys

setting = sys.argv[1]
data_dir = sys.argv[2]
output_dir = 'l2m'

SEP = '[sep]'
ALIGN = '[align]'
beam = 5

input_lyric_file = f'{data_dir}/para/valid.lyric'
generate_melody_file = '../result/%s'%setting

generated_file = True
get_last = True
find_structure = True
cut_exceed_sent = True


def get_pitch_duration_structure(note_seq):
    seq = []
    
    #遍历寻找pitch-duration的结构
    #当有不合法情况出现时，找最后一个pitch和第一个duration，保证其相邻
    #p1 d1 p2 p3 d2 p4 d3-> p1 d1 p3 d1 p4 d3
    #p1 d1 p2 d2 d3 p3 d4-> p1 d1 p2 d2 p3 d4
    #p1 d1 p2 p3 d2 d3 p4 d4 -> p1 d1 p3 d2 p4 d4
    
    i = 0
    while (i<len(note_seq)):
        if note_seq[i] > 128:
            #Duration
            i += 1
            
        else:
            #Pitch
            if i+1>=len(note_seq):
                #No Duration Followed
                break
            if note_seq[i+1] <= 128:
                #Followed by a pitch
                i += 1
                continue
        
            
            #Here trans back to str for bleu calculate
            pitch = str(note_seq[i])
            duration = str(note_seq[i+1])
            
            seq.append(pitch)
            seq.append(duration)
            i += 2
    return seq

def separate_sentences(x,find_structure = False):
    lst = x.copy()
    sep_positions = [i for i,x in enumerate(lst) if x==SEP]
    sep_positions.insert(0,-1)

    ret = []
    for i in range(len(sep_positions)-1):
        sent = lst[sep_positions[i]+1:sep_positions[i+1]] #SZH: not include sep token
        if find_structure:
            sent = list(map(int, sent))
            sent = get_pitch_duration_structure(sent)
        ret.append(sent)
    return ret

def get_songs(file, generated_file = False, get_last = False, find_structure = False, cut_exceed_sent=False):
    """
    Get Last : Whether include the last tokens if the sequence not ends with a seperation token
    """
    with io.open(input_lyric_file, 'r') as f:
        input_lyrics= f.readlines()

    input_lyrics = list(map(lambda x:x.rstrip('\n').split(' '), input_lyrics))
    input_lyrics_sent_num = list(map(lambda x:x.count(SEP), input_lyrics))
    
    with io.open(file, 'r') as f:
        melody_lines = f.readlines()
        if generated_file:
            melody_lines = list(filter(lambda x:x.startswith('H-'), melody_lines))
            if len(melody_lines) == len(input_lyrics) * beam:
                melody_lines.sort( key=lambda x:(int(x.split('\t')[0].split('-')[1]), -float(x.split('\t')[1]) ) )
                melody_lines = [ x for i,x in enumerate(melody_lines) if i%beam == 0 ]
            else:    
                melody_lines.sort( key=lambda x:int(x.split('\t')[0].split('-')[1]) )
    melody_lines = list(map(lambda x:x.rstrip('\n').split('\t')[-1], melody_lines))
            
    # print(len(melody_lines), len(input_lyrics))
    assert len(melody_lines) == len(input_lyrics)
    
    melody_seqs = list(map(lambda x:x.rstrip('\n').split(' '),melody_lines))
    melody_seqs = [ch for ch in melody_seqs if ch != ALIGN]
    
    for i in range(len(melody_seqs)):
        melody_seqs[i] = list(filter(lambda x:x.isdigit() or x==SEP, melody_seqs[i]))
        
    if get_last:
        for i in range(len(melody_seqs)):
            if melody_seqs[i][-1] != SEP:
                melody_seqs[i].append(SEP)
          
    # 分句子，同时find structure
    melody_seq_sents = list(map(lambda x:separate_sentences(x,find_structure=find_structure), melody_seqs))

    
    # 把句子组合回整首歌，同时切除过多的句子
    return_list = []
    for i,sent_seq in enumerate(melody_seq_sents):
        if cut_exceed_sent and len(sent_seq) > input_lyrics_sent_num[i]:
            sent_seq = sent_seq[0:input_lyrics_sent_num[i]]
        cur_song_return = []

        for j,sent in enumerate(sent_seq):
            cur_song_return.extend(sent)
            cur_song_return.append(SEP)
        return_list.append(cur_song_return)
    return return_list
            


output_melodies = get_songs(generate_melody_file, generated_file, get_last, find_structure, cut_exceed_sent)

with io.open(input_lyric_file,'r') as f:
    input_lyrics= f.readlines()
input_lyrics = list(map(lambda x:x.rstrip('\n').split(' '), input_lyrics))
input_lyrics_sent_num = list(map(lambda x:x.count(SEP), input_lyrics))

output_melodies_sent_num = list(map(lambda x:x.count(SEP), output_melodies))
list(zip(input_lyrics_sent_num, output_melodies_sent_num))


if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)



for i,x in enumerate(zip(input_lyrics,output_melodies)):
    cur_folder = os.path.join(output_dir, str(i))
    os.makedirs(cur_folder)
    with io.open(os.path.join(cur_folder,'train.lyric'),'w') as f:
        f.write(" ".join(x[0]))
    with io.open(os.path.join(cur_folder,'train.melody'),'w') as f:
        f.write(" ".join(x[1]))




