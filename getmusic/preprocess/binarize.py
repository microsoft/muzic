import multiprocessing as mp
import random
from tqdm import tqdm
import numpy as np
import sys
import os
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from getmusic.utils.midi_config import *
from getmusic.data.indexed_datasets import IndexedDatasetBuilder

prog_to_abrv = {'80':'M', '32':'B', '128':'D', '25':'G', '0':'P', '48':'S',}
inst_to_row = {'80':0, '32':1, '128':2, '25':3, '0':4, '48':5, '129':6}
root_dict = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
kind_dict = {'null': 0, 'm': 1, '+': 2, 'dim': 3, 'seven': 4, 'maj7': 5, 'm7': 6, 'm7b5': 7}
root_list = list(root_dict.keys())
kind_list = list(kind_dict.keys())


pos_in_bar = beat_note_factor * max_notes_per_bar * pos_resolution

figure_size = bar_max * pos_in_bar
    
def oct_to_rep(line):
    
    no_empty_tracks = {'80':0,'32':0,'128':0,'25':0,'0':0,'48':0}
    main_num = line.count('<2-80>')
    string_num = line.count('<2-48>')
    drum_num = line.count('<2-128>')

    oov = 0
    inv = 0
    
    if '<2-80>' not in line:
        return None

    f = ('<2-0>' in line) + ('<2-25>' in line) + ('<2-32>' in line) + ('<2-48>' in line) + ('<2-128>' in line)

    if f == 0:
        return None


    tmp = line.strip().split('<0-')[1:]
    encoding = []
    for item in tmp:
        tmp2 = item.strip()[:-1].split('> <')
        encoding.append([int(tmp2[0])] + [int(i[2:]) for i in tmp2[1:]])
    
    chord_list = []

    datum = pad_index * np.ones([14, 1 + figure_size],dtype=float)

    idx = 0
    while idx != len(encoding) - 1:
        e = encoding[idx]

        bar = e[0]
        pos = e[1]
        inst = e[2]
        pitch = e[3]
        tempo = e[7]
        
        assert e[6] == 6 # ts = 4/4

        if e[2] == 129:
            row = inst_to_row[str(inst)]
            r = root_list[e[3]]
            k = kind_list[e[4]]
            datum[2 * row][pos_in_bar * bar + pos : pos_in_bar * (bar + 1) + pos] = tokens_to_ids[r]
            datum[2 * row + 1][pos_in_bar * bar + pos : pos_in_bar * (bar + 1) + pos] = tokens_to_ids[k]
            idx += 1
            continue
        
        if tempo == 0:
            return None
        
        dur = e[4]
        if e[4] == 0:
            e[4] = 1

        chord_list = [str(e[3])]

        for f_idx in range(idx + 1, len(encoding)):
            if (encoding[f_idx][0] == bar) and (encoding[f_idx][1] == pos) and (encoding[f_idx][2] == inst):
                if encoding[f_idx][3] != pitch:
                    chord_list.append(str(encoding[f_idx][3]))
                    pitch = encoding[f_idx][3]
            else:
                break
        
        idx = max(idx + 1, f_idx)

        row = inst_to_row[str(inst)]
        dur = tokens_to_ids['T'+str(e[4])]
        
        chord_string = ' '.join(chord_list)
        token = prog_to_abrv[str(inst)] + chord_string

        if token in tokens_to_ids:
            pitch = tokens_to_ids[token]
            no_empty_tracks[str(e[2])] = 1
            assert (dur < pad_index) and (pitch > pad_index), 'pitch index is {} and dur index is {}'.format(pitch, dur)
            datum[2 * row][pos_in_bar * bar + pos] = pitch
            datum[2 * row + 1][pos_in_bar * bar + pos] = dur
            inv += 1
        else:
            oov += 1

    if sum(no_empty_tracks.values()) > 1 and no_empty_tracks['80'] == 1: # 伴奏 + 主旋律
        
        assert (datum == pad_index).sum() != 14 * (pos_in_bar * bar_max+1)

        pad_in_a_row = (datum == pad_index).sum(-1)
        for row_id in range(6):     
            if pad_in_a_row[2 * row_id] == figure_size + 1:
                datum[2 * row_id] = empty_index
                datum[2 * row_id + 1] = empty_index

        datum[:,pos_in_bar * bar + pos + 1:] = empty_index
        datum[:,-1] = tempo

        return datum
    else:
        return None

if __name__ == "__main__":
  
    tokens_to_ids = {} 
    
    ids_to_tokens = [] 
    dict_path = sys.argv[1]
    with open(dict_path,'r') as f:
        tokens = f.readlines()
        for id, token in enumerate(tokens):
            token, freq = token.strip().split('\t')
            tokens_to_ids[token] = id
            ids_to_tokens.append(token)
        pad_index = tokens_to_ids['<pad>']
        empty_index = len(ids_to_tokens)
        print('{} tokens in dictionary'.format(len(ids_to_tokens)))
    
    oct_path = sys.argv[2]
    binary_data_dir = sys.argv[3]
    os.makedirs(binary_data_dir, exist_ok=True)

    num_valid_split = 10
    
    with open(oct_path ,'r') as f:
        all_midi_fns = f.readlines()

    for split in ['valid', 'train']:
        p = mp.Pool(int(os.getenv('N_PROC', os.cpu_count())))
        futures = []
        midi_fns = all_midi_fns[num_valid_split:] if split == 'train' \
            else all_midi_fns[:num_valid_split]
        print(f"| #{split} set: {len(midi_fns)}")
        ds_builder = IndexedDatasetBuilder(f'{binary_data_dir}/{split}')
        for midi_fn in midi_fns:
            futures.append(p.apply_async(oct_to_rep, args=[midi_fn]))
        p.close()

        num = 0
        for f in tqdm(futures):
            item = f.get()
            if item is None:
                continue
            num +=1
            ds_builder.add_item(item)
        ds_builder.finalize()
        p.join()
        np.save(f'{binary_data_dir}/{split}_length.npy'.format(split), num)
        print('{} set has {} reps'.format(split, num))
