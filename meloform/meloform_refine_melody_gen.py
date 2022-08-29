# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import sys, os, json
from utils import enc_vel, enc_ts, enc_tpo, encoding_to_midi
from meloform.meloform_transformer import TransformerMeloFormModel

def load_data(data_path):
    with open(data_path + '.template', 'r') as fr:
        for line in fr:
            source = line.strip()
    with open(data_path + '.melody', 'r') as fw:
        for line in fw:
            target = line.strip()
    return source, target

def write_to_txt(fn, hypos, source, target, subset, select_idx=None):
    if select_idx is None:
        with open(fn, 'w') as f:
            f.write('TID-' + '\t' + subset + '\n')
            f.write('S-' + '\t' + source + '\n')
            f.write('T-' + '\t' + target + '\n')
            for hypo in hypos:
                output = hypo['tokens']
                output = [label_dict[x] for x in output]
                output = ' '.join(output)

                f.write('H-' + '\t' + str(hypo['score'].item()) + '\t' + output + '\n')
    else:
        with open(fn, 'w') as f:
            f.write('TID-' + '\t' + subset + '\n')
            f.write('S-' + '\t' + source + '\n')
            f.write('T-' + '\t' + target + '\n')
            hypo = hypos[select_idx]
            output = hypo['tokens']
            output = [label_dict[x] for x in output]
            output = ' '.join(output)

            f.write('H-' + '\t' + str(hypo['score'].item()) + '\t' + output + '\n')

def generate_raw(fn, song_id, phrase, out_dir):
    out_dir = os.path.join(out_dir, song_id, phrase)
    os.makedirs(out_dir, exist_ok=True)
    subset = f'test-{song_id}-melody-{phrase}-update'
    source, target = load_data(os.path.join(fn, subset))
    tokenized_sentences = [model.encode(source)]
    batched_hypos = model.generate(tokenized_sentences,
                                     sampling=True,
                                     sampling_topk=topk,
                                     sampling_topp=topp,
                                     temperature=temperature,
                                     max_len_a=0,
                                     max_len_b=5000,
                                     min_len=4,
                                     verbose=True,
                                     beam=1,
                                     need_attn=True)

    hypos = batched_hypos[0]

    select_idx = -1
    max_prob = -100
    for i in range(len(hypos)):
        if hypos[i]['score'].item() > max_prob:
            max_prob = hypos[i]['score'].item()
            select_idx = i

    os.system('mkdir {}'.format(out_dir))
    os.system('mkdir {}'.format(os.path.join(out_dir, 'fig_attn')))
    write_to_txt(os.path.join(out_dir, subset) + '.txt', hypos, source, target, subset, select_idx)

    hypo = hypos[select_idx]
    output = hypo['tokens']
    output = [label_dict[x] for x in output]
    output = ' '.join(output)
    return source, target, output

# ############################################################ infer phrase melody ###################################################

def extract_target(l):
    res = []
    sent = []
    for v in l:
        if v == '[sep]':
            res.append(sent)
            sent = []
        else:
            sent.append(v)
    if len(sent) > 0:
        if sent[-1] == '</s>':
            sent = sent[:-1]
        res.append(sent)
    return res

def extract_src(res):
    sent = []
    for v in res:
        if v != '[sep]':
            sent.append(v)
    return sent

def fix(items):
    tmp = []
    target_tokens = ['Bar', 'Pos', 'Pitch', 'Dur']
    i = 0
    for item in items:
        if item.split('_')[0] == target_tokens[i]:
            tmp.append(item)
            i = (i + 1) % len(target_tokens)
    return tmp

def adapt_e(e, min_pitch, max_pitch):
    tmp = [list(i) for i in e]
    last_pos = 0
    for i in range(len(tmp)):
        note = tmp[i]

        # 16th note
        if note[1] % 2 == 1 and last_pos <= (16 * note[0] + note[1] - 1):
            note[1] -= 1
        if note[4] != 1 and (note[1] + note[4]) % 2 == 1:
            note[4] -= 1
        if last_pos >= 16 * note[0] + note[1]:
            tmp[i-1][4] -= last_pos - (16 * note[0] + note[1])
        last_pos = 16 * note[0] + note[1] + note[4]
        tmp[i] = note

    tmp = [tuple(i) for i in tmp]
    return tmp

tempo = 120
def convert_to_midi_obj(sent, min_pitch=52, max_pitch=80):
    enc = fix(sent)
    e = list(map(lambda x: int(''.join(filter(str.isdigit, x))), enc))
    e = [(e[i], e[i + 1], 0, e[i + 2], e[i + 3], enc_vel(127),
            enc_ts((4, 4)), enc_tpo(tempo)) for i in range(0, len(e) // 4 * 4, 4)]

    min_bar = min([i[0] for i in e])
    e = [tuple(k - min_bar if j == 0 else k for j,
                k in enumerate(i)) for i in e]
    e.sort()
    e = [tuple(i) for i in e]
    e = adapt_e(e, min_pitch, max_pitch)
    midi_obj = encoding_to_midi(e)
    return midi_obj

def generate_midi(root_dir, song_id, phrase, output, prefix=''):
    os.makedirs(os.path.join(root_dir, song_id, phrase), exist_ok=True)

    res = output.split(' ')
    res_list = extract_target(res)
    for sent_id, sent in enumerate(res_list):
        midi_obj = convert_to_midi_obj(sent)
        midi_obj.dump(os.path.join(midi_out_dir, song_id, phrase, prefix + '-' + str(sent_id) + '.mid'))


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

def get_feature(sample, st_idx, feature=''):
    counter = 1
    while counter < 10:
        if feature in sample[st_idx - counter].split('_')[0]:
            res = int(sample[st_idx - counter].split('_')[1])
            return res
        counter += 1
    return None

SEP='[sep]'
def reorder_bar(sample):

    offset = 0
    new_sample = []
    prev_idx = 0
    for i, item in enumerate(sample):
        new_item = item
        if 'Bar' in item:
            cur_idx = int(item.split('_')[1])

            if i - 1 > 0 and sample[i - 1] == SEP:
                
                offset += prev_idx + 1
            
            new_item = 'Bar_' + str(cur_idx + offset)
            prev_idx = cur_idx
        new_sample.append(new_item)
    return new_sample
    
def generate_midi_combine(midi_out_dir, song_id, phrase, source, output, prefix=''):
    phrase_position_dicts = get_phrase_position_dicts(os.path.join(template_dir, 'template.json'))
    tgt_send_ids = phrase_position_dicts[phrase]
    # src + res
    src = source.split(' ')
    res = output.split(' ')
    parts = extract_target(res)
    sep_positions = [
            i for i, x in enumerate(src) if x == '[sep]'
    ]
    sep_positions.insert(0, -1)

    segments = []
    for sent_id in range(len(sep_positions) - 1):
        segments.append(src[sep_positions[sent_id] + 1:sep_positions[sent_id + 1] + 1])

    for j in range(len(tgt_send_ids)):
        if j >= len(parts):
            segments[tgt_send_ids[j]] = parts[0] + ['[sep]']
        else:
            segments[tgt_send_ids[j]] = parts[j] + ['[sep]']

    segments = [item for sublist in segments for item in sublist]

    sent_order = reorder_bar(segments)
    sent_order = extract_src(sent_order)
    midi_obj = convert_to_midi_obj(sent_order)

    midi_obj.dump(os.path.join(midi_out_dir, song_id, phrase, prefix + '.mid'))


if __name__ == '__main__':
    data_dir = sys.argv[1]
    song_id = sys.argv[5]
    phrase = sys.argv[6]
    topk = int(sys.argv[7])
    topp = float(sys.argv[8])
    temperature = float(sys.argv[9])

    data_dir = os.path.join(data_dir, song_id)
    out_dir = sys.argv[4]
    os.system('mkdir {}'.format(out_dir))
    template_dir = f'{data_dir}/template'

    model = TransformerMeloFormModel.from_pretrained(
        model_name_or_path=sys.argv[2],
        checkpoint_file=sys.argv[3],
        data_name_or_path='../data/train/processed/processed_para',
        beam=5,
        )

    label_dict = model.task.target_dictionary

    model.cuda()
    model.eval()

    midi_out_dir = os.path.join(out_dir, 'out_midi')
    raw_out_dir = os.path.join(out_dir, 'out_raw')
    source, target, output = generate_raw(data_dir, song_id, phrase, raw_out_dir)

    # generate output and target
    print('################ generating refined phrases. ################')
    generate_midi(midi_out_dir, song_id, phrase, output, 'seg')
    print('################ generating target phrases. ################')
    generate_midi(midi_out_dir, song_id, phrase, target, 'tgt')
    print('################ generating melody with refined phrases. ################')
    generate_midi_combine(midi_out_dir, song_id, phrase, source, output, 'src_res')

