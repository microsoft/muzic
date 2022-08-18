import random
from matplotlib.pyplot import bar
import numpy as np
from tqdm import tqdm
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from find_chorus import Find_Chorus
from fairseq.models.transformer_lm import TransformerLanguageModel
import argparse

def check_not(notes_string):    
    tmp = [0] * 128
    notes = notes_string.split(' ')[3::5]
    for idx in range(len(notes)):    
        tmp[int(notes[idx][6:])] = 1
    if (1 < len(notes) <= 3 and sum(tmp) == 1) or (len(notes) >= 4 and sum(tmp) < 3):
        return False
            
    return True

custom_lm = TransformerLanguageModel.from_pretrained('../music-ckps/', 'checkpoint_best.pt', tokenizer='space', batch_size=4096).cuda()

parser = argparse.ArgumentParser(description='none.')
parser.add_argument('notes_path')
config = parser.parse_args()
notes_path = config.notes_path
wc_path = notes_path.split('.')[0] + '_chorus.notes'
wv_path = notes_path.split('.')[0] + '_verse.notes'

embedding_path = 'embedding.txt'

if 'maj' in notes_path:
    MAJOR = 1
else:
    MAJOR = 0

print('loading embedding.')
with open(embedding_path) as notes_file:
    melodies = notes_file.readlines()
    embed = np.zeros([108, 100])
    for melody in melodies:
        w, str_vec = melody.split(' ', 1)
        ptype = int(w)
        vec = np.fromstring(str_vec, sep=" ")
        embed[ptype] = vec
print('embedding loaded.')

with open(notes_path,'r') as r, open(wc_path,'w') as wc, open(wv_path,'w') as wv:
    songs = r.readlines()
    num = 0
    for song in tqdm(songs):
        # song = 'A 0 X X X A 0 X X X A 0 X X X A 0 X X X A 1 X X X A 2 X X X A 2 X X X A 3 X X X A 3 X X X A 4 X X X A 4 X X X'
        notes = song.strip().split(' ')
        matrics = []
        cur_bar = 0
        length = len(notes) // 5
        for i in range(length):
            pitch = int(notes[5 * i + 3][6:])
            matrics.append(embed[pitch])

        SSM = cosine_similarity(matrics)
        chorus = Find_Chorus(SSM, length)
        chorus_start = 0
        chorus_end = 0
        if chorus is not None and chorus.end - chorus.start > 20:  # 副歌
            chorus_end =chorus.end
            chorus_start = chorus.start
        else: 
            continue


        start_bar = 0
        tmp = ''
        last_gtmp = ''
        cnt = 0
        CHORUS = 0

        for i in range(len(notes) // 5):
            if i == chorus_start:
                CHORUS = 1
            if i == chorus_end:
                CHORUS = 0

            cadence = notes[5 * i]
            bar_idx = int(notes[5 * i + 1][4:])
            pos = notes[5 * i + 2] # // pos_resolution
            pitch = notes[5 * i + 3]
            dur = notes[5 * i + 4]

            assert bar_idx >= start_bar

            if bar_idx - start_bar >= 2:
                if check_not(tmp):
                    g_tmp = ''
                    generated_melody = custom_lm.sample(tmp).split(' ')[cnt * 5:]
                    if len(generated_melody) >= 5:
                        start_bar__ = int(generated_melody[1][4:])
                        for idx in range(len(generated_melody) // 5):
                            bar_idx__ = int(generated_melody[idx * 5 + 1][4:])
                            if bar_idx__ - start_bar__ >= 2:
                                if last_gtmp == tmp:
                                    num += 1
                                else:
                                    if CHORUS:
                                        wc.write(g_tmp)
                                        wc.write('\n')
                                    else:
                                        wv.write(g_tmp)
                                        wv.write('\n')
                                last_gtmp = g_tmp
                                break
                            else:
                                g_tmp += '{} {} {} {} {} '.format(generated_melody[idx * 5], generated_melody[idx * 5 + 1], generated_melody[idx * 5 + 2], generated_melody[idx * 5 + 3], generated_melody[idx * 5 + 4])
                            
                else:
                    continue

                tmp = ''  
                start_bar = bar_idx
                cnt = 0
          
            if bar_idx - start_bar < 2:
                tmp += '{} bar_{} {} {} {} '.format(cadence, bar_idx, pos, pitch, dur)
                cnt += 1

    print(num)
            
            
        
        
        
        
        
        
        
        