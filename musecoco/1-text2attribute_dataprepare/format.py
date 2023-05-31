import os, pickle, json, random
from tqdm import tqdm
from midi_data_extractor.attribute_unit import convert_value_dict_into_unit_dict
import midiprocessor as mp
from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy

attributes = [
    'I1s2',
    'R1',
    'R3',
    'S2s1',
    'S4',
    'B1s1',
    'TS1s1',
    'K1',
    'T1s1',
    'P4',
    'EM1',
    'TM1'
]
instruments = [
    'piano',
    'keyboard',
    'percussion',
    'organ',
    'guitar',
    'bass',
    'violin',
    'viola',
    'cello',
    'harp',
    'strings',
    'voice',
    'trumpet',
    'trombone',
    'tuba',
    'horn',
    'brass',
    'sax',
    'oboe',
    'bassoon',
    'clarinet',
    'piccolo',
    'flute',
    'pipe',
    'synthesizer', 
    'ethnic instruments',
    'sound effects',
    'drum']
artist_label_to_artist_name = {
    'beethoven': 'Beethoven',
    'mozart': 'Mozart',
    'chopin': 'Chopin',
    'schubert': 'Schubert',
    'schumann': 'Schumann',
    'bach-js': 'Bach',
    'haydn': 'Haydn',
    'brahms': 'Brahms',
    'Handel': 'Handel',
    'tchaikovsky': 'Tchaikovsky',
    'mendelssohn': 'Mendelssohn',
    'dvorak': 'Dvorak',
    'liszt': 'Liszt',
    'stravinsky': 'Stravinsky',
    'mahler': 'Mahler',
    'prokofiev': 'Prokofiev',
    'shostakovich': 'Shostakovich',
}
genre_label_to_name = {
    'New Age': "new age",
    'Electronic': "electronic",
    'Rap': 'rap',
    'Religious': 'religious',
    'International': 'international',
    'Easy_Listening': 'easy listening',
    'Avant_Garde': 'avant-garde',
    'RnB': 'RnB',
    'Latin': 'Latin',
    'Children': 'children',
    'Jazz': 'jazz',
    'Classical': 'classical',
    'Comedy_Spoken': 'comedy',
    'Pop_Rock': 'pop',
    'Reggae': 'reggae',
    'Stage': 'stage',
    'Folk': 'folk',
    'Blues': 'blues',
    'Vocal': 'vocal',
    'Holiday': 'holiday',
    'Country': 'country',
    "Symphony": 'symphony'
}



if __name__ == "__main__":
    print('Load data...')
    attvector = pickle.load(open('test_refined/0_test.bin', 'rb'))
    print('Done')

    encoder = mp.MidiEncoder('REMIGEN')

    I1s2 = ['I1s2_'+ '_'.join(ins.split(' ')) for ins in instruments]
    S4 = ['S4_' + '_'.join(genre.lower().split(' ')) for genre in genre_label_to_name.keys()]


    for idx in tqdm(range(len(attvector))):
        unit_dict = convert_value_dict_into_unit_dict(attvector[idx]['use_info_dict'], encoder=encoder)
        attvector[idx]['labels'] = OrderedDict()
        for k,v in attvector[idx]['use_info_dict'].items():
            # print(k)
            if v==None or v==(None, None):  
                if k == "I1s2":
                    allvector = unit_dict[k].get_vector(use=False)
                    for i, ins in enumerate(I1s2):
                        attvector[idx]['labels'][ins] = allvector[i]
                elif k == "S4":
                    allvector = unit_dict[k].get_vector(use=False)
                    for i, genre in enumerate(S4):
                        attvector[idx]['labels'][genre] = allvector[i]
                else:
                    attvector[idx]['labels'][k] = unit_dict[k].get_vector(use=False)
            else:
                if k=='I1s2':
                    allvector = unit_dict[k].get_vector(use=True, use_info=v)
                    for i, ins in enumerate(I1s2):
                        attvector[idx]['labels'][ins] = allvector[i]
                elif k=="S4":
                    allvector = unit_dict[k].get_vector(use=True, use_info=v)
                    for i, genre in enumerate(S4):
                        attvector[idx]['labels'][genre] = allvector[i]
                elif k=="I4" and v[1]==False:
                    attvector[idx]['labels'][k] = unit_dict[k].get_vector(use=False)
                else:
                    attvector[idx]['labels'][k] = unit_dict[k].get_vector(use=True) 
    
    att_key = list(attvector[0]['labels'].keys())
    json.dump(att_key, open('att_key.json','w'))

    testjson = []
    for idx in tqdm(range(len(attvector))):
        ins = {}
        ins['text'] = deepcopy(attvector[idx]['text'])
        ins['labels'] = []
        for att in att_key:
            ins['labels'].append(deepcopy(attvector[idx]['labels'][att]))
        testjson.append(deepcopy(ins))
    json.dump(testjson, open('test.json','w'))
