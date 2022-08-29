# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import sys, os, json

############################################################# Replace phrase ###################################################
def get_phrase_position_dicts(path):
    f = open(path)
    template = json.load(f)
    f.close()
    phrase_structure = template['phrase structure']
    phrase_structure = [item for sublist in phrase_structure for item in sublist]
    dicts = {}
    for i, phrase in enumerate(phrase_structure):
        if phrase not in dicts:
            dicts[phrase] = [i]
        else:
            dicts[phrase].append(i)
    return dicts

if __name__ == '__main__':

    src_dir = sys.argv[1]
    song_id = sys.argv[2]
    src_phrase = sys.argv[3]
    tgt_phrase = sys.argv[4]
    tgt_dir = sys.argv[5]

    src_subset = f'test-{song_id}-melody-{src_phrase}-update'
    tgt_subset = f'test-{song_id}-melody-{tgt_phrase}-update'

    tplt_dir = f'{src_dir}/{song_id}/template'
    phrase_position_dicts = get_phrase_position_dicts(os.path.join(tplt_dir, 'template.json'))
    rpl_send_ids = phrase_position_dicts[tgt_phrase]


    with open(os.path.join(src_dir, song_id, src_subset + '.template'), 'r') as f:
        src = f.readline().strip().split()

    with open(os.path.join(tgt_dir, 'out_raw', song_id, tgt_phrase, tgt_subset + '.txt'), 'r') as f:
        for line in f:
            if line[:2] == 'H-':
                rpls_read = line.strip().split('\t')[2].split()

    rpls = []
    tmp = []
    for x in rpls_read:
        if x == '[sep]' or x == '</s>':
            rpls.append(tmp)
            tmp = []
        else:
            tmp.append(x)


    if len(rpls) < len(rpl_send_ids):
        print(f'{song_id}-{src_phrase} don\'t have enough sentences.')
        for i in range(len(rpl_send_ids) - len(rpls)):
            rpls.append(rpls[0])

    assert(len(rpls) >= len(rpl_send_ids))

    sep_positions = [
                i for i, x in enumerate(src) if x == '[sep]'
    ]
    sep_positions.insert(0, -1)

    for i, rpl in enumerate(rpls):
        rpl_process = rpl
        rpls[i] = rpl_process

    segments = []
    for sent_id in range(len(sep_positions) - 1):
        segments.append(src[sep_positions[sent_id] + 1:sep_positions[sent_id + 1] + 1])

    new_src = []
    for i, rpl_id in enumerate(rpl_send_ids):
        segments[rpl_id] = rpls[i] + ['[sep]']

    for seg in segments:
        new_src.extend(seg)


    with open(os.path.join(src_dir, src_subset + '.template'), 'w') as f:
        f.write(' '.join(new_src) + '\n')