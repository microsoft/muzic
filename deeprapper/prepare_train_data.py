# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import jieba
import json
import numpy as np
import os
import re
import random
import pypinyin
import shutil
from pypinyin import lazy_pinyin, Style
from tqdm.auto import tqdm


def get_shuffled_samples(a, b, c, d, e):
    length = len(a)
    flag = [1, 1, 1, 1]
    if b == []:
        b = np.zeros(length)
        flag[0] = 0
    if c == []:
        c = np.zeros(length)
        flag[1] = 0
    if d == []:
        d = np.zeros(length)
        flag[2] = 0
    if e == []:
        e = np.zeros(length)
        flag[3] = 0
    samples = list(zip(a, b, c, d, e))
    random.shuffle(samples)
    a, b, c, d, e = zip(*samples)
    if flag[0] == 0:
        b = None
    if flag[1] == 0:
        c = None
    if flag[2] == 0:
        d = None
    if flag[3] == 0:
        e = None
    return a, b, c, d, e        


def remove_prefix(text, prefix):
    while text.startswith(prefix):
        text = text[len(prefix):]
    return text


def remove_suffix(text, suffix):
    while text.endswith(suffix):
        text = text[:-len(suffix)]
    return text


def segment_text(lines):
#     jieba.enable_paddle()
#     l = ' '.join(jieba.lcut(lines[0], use_paddle=True))
#     print(l)
    all_len = len(lines)
    k = 0
    for i in range(all_len):
        try:
            l = ' '.join(jieba.lcut(lines[i]))
            lines[i] = l
        except:
            k += 1
            print(l)
#     lines = [jieba.lcut(l) for l in lines]
#     return [' '.join(l) for l in lines]
    print(f'{k}/{all_len}')
    return lines


def build_files_separate(num_pieces,
                         stride,
                         min_length,
                         lines=None, 
                         finals=None,
                         sentences=None,
                         pos=None,
                         beats=None,
                         tokenized_data_path=None, 
                         finalized_data_path=None,
                         sentenced_data_path=None,
                         posed_data_path=None,
                         beated_data_path=None,
                         full_tokenizer=None, 
                         full_finalizer=None,
                         full_sentencer=None,
                         full_poser=None,
                         full_beater=None,
                         enable_final=False,
                         enable_sentence=False,
                         enable_pos=False,
                         enable_beat=False,
                         segment=False):
    print('Start tokenizing..')
    assert len(lines) == len(finals) == len(sentences)
    if segment:
        lines = segment_text(lines)
    path = tokenized_data_path.rsplit('/', 1)[0]
    if not os.path.exists(path):
        os.mkdir(path)
    print(f'#lines: {len(lines)}')
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path) 
    if enable_final:
        print(f'#finals: {len(finals)}')
        if not os.path.exists(finalized_data_path):
            os.mkdir(finalized_data_path)
    if enable_sentence:
        print(f'#sentences: {len(sentences)}')
        if not os.path.exists(sentenced_data_path):
            os.mkdir(sentenced_data_path)
    if enable_pos:
        print(f'#pos: {len(pos)}')
        if not os.path.exists(posed_data_path):
            os.mkdir(posed_data_path)
    if enable_beat:
        print(f'#beats: {len(beats)}')
        if not os.path.exists(beated_data_path):
            os.mkdir(beated_data_path)
    
    all_len = len(lines)
    
    for k in range(num_pieces):
        max_length = stride - 2 
        print(max_length)
        
        for i in range(len(lines)):
            line = lines[i]
            if len(line) > min_length:
                line = full_tokenizer.tokenize(line)
                line = full_tokenizer.convert_tokens_to_ids(line)
                line_length = len(line)
                skip = full_tokenizer.convert_tokens_to_ids('[SKIP]')
                skips = [skip] * max_length
                if line_length >= max_length:
                    line = line[0:max_length]
                else:
                    skips[0:line_length] = line[0:line_length]
                    line = skips
    
                if enable_final:
                    final = finals[i]
                    final = full_finalizer.tokenize(final)
                    final = full_finalizer.convert_tokens_to_ids(final) 
                    skip = full_finalizer.convert_tokens_to_ids('[SKIP]')
                    skips = [skip] * max_length
                    if line_length >= max_length:
                        final = final[0:max_length]
                    else:
                        skips[0:line_length] = final[0:line_length]
                        final = skips
                    assert len(final) == len(line)
                           
                if enable_sentence:
                    sentence = sentences[i]
                    sentence = full_sentencer.tokenize(sentence)
                    sentence = full_sentencer.convert_tokens_to_ids(sentence)
                    skip = full_sentencer.convert_tokens_to_ids('[SKIP]')
                    skips = [skip] * max_length
                    if line_length >= max_length:
                        sentence = sentence[0:max_length]
                    else:
                        skips[0:line_length] = sentence[0:line_length]
                        sentence = skips
                    assert len(sentence) == len(line)
                    
                if enable_pos:
                    p = pos[i]
                    p = full_poser.tokenize(p)
                    p = full_poser.convert_tokens_to_ids(p)
                    skip = full_poser.convert_tokens_to_ids('[SKIP]')
                    skips = [skip] * max_length
                    if line_length >= max_length:
                        p = p[0:max_length]
                    else:
                        skips[0:line_length] = p[0:line_length]
                        p = skips
                    assert len(p) == len(line)
                
                if enable_beat:
                    beat = beats[i]
                    beat = full_beater.tokenize(beat)
                    beat = full_beater.convert_tokens_to_ids(beat)
                    skip = full_beater.convert_tokens_to_ids('[SKIP]')
                    skips = [skip] * max_length
                    if line_length >= max_length:
                        beat = beat[0:max_length]
                    else:
                        skips[0:line_length] = beat[0:line_length]
                        beat = skips
                    assert len(beat) == len(line)
                
                lines[i] = line
                if enable_final:
                    finals[i] = final
                if enable_sentence:
                    sentences[i] = sentence
                if enable_pos:
                    pos[i] = p
                if enable_beat:
                    beats[i] = beat
        
        full_line, full_final, full_sentence, full_pos, full_beat = [], [], [], [], []
        for i in range(len(lines)):
            mask = full_tokenizer.convert_tokens_to_ids('[MASK]')
            clss = full_tokenizer.convert_tokens_to_ids('[CLS]')
            full_line.append(mask)  # 文章开头添加MASK表示文章开始
            full_line.extend(lines[i])
            full_line.append(clss)  # 文章之间添加CLS表示文章结束
            
            if enable_final:
                mask = full_finalizer.convert_tokens_to_ids('[MASK]')
                clss = full_finalizer.convert_tokens_to_ids('[CLS]')
                full_final.append(mask)  # 文章开头添加MASK表示文章开始
                full_final.extend(finals[i])
                full_final.append(clss)  # 文章之间添加CLS表示文章结束
            
            if enable_sentence:
                mask = full_sentencer.convert_tokens_to_ids('[MASK]')
                clss = full_sentencer.convert_tokens_to_ids('[CLS]')
                full_sentence.append(mask)  # 文章开头添加MASK表示文章开始
                full_sentence.extend(sentences[i])
                full_sentence.append(clss)  # 文章之间添加CLS表示文章结束
                
            if enable_pos:
                mask = full_poser.convert_tokens_to_ids('[MASK]')
                clss = full_poser.convert_tokens_to_ids('[CLS]')
                full_pos.append(mask)  # 文章开头添加MASK表示文章开始
                full_pos.extend(pos[i])
                full_pos.append(clss)  # 文章之间添加CLS表示文章结束    
            
            if enable_beat:
                mask = full_beater.convert_tokens_to_ids('[MASK]')
                clss = full_beater.convert_tokens_to_ids('[CLS]')
                full_beat.append(mask)  # 文章开头添加MASK表示文章开始
                full_beat.extend(beats[i])
                full_beat.append(clss)  # 文章之间添加CLS表示文章结束 
        
        if enable_final:
            assert len(full_line) == len(full_final), f'line: {len(full_line)}, final: {len(full_final)}'
        if enable_sentence:
            assert len(full_line) == len(full_sentence), f'line: {len(full_line)}, sentence: {len(full_sentence)}'
        if enable_pos:
            assert len(full_line) == len(full_pos), f'line: {len(full_line)}, pos: {len(full_pos)}'
        if enable_beat:
            assert len(full_line) == len(full_beat), f'line: {len(full_line)}, beat: {len(full_beat)}'
        
        with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
            for idx in full_line:
                f.write(str(idx) + ' ')
        
        if enable_final:
            with open(os.path.join(finalized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_final:
                    f.write(str(idx) + ' ')
        
        if enable_sentence:
            with open(os.path.join(sentenced_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_sentence:
                    f.write(str(idx) + ' ')
                    
        if enable_pos:
            with open(os.path.join(posed_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_pos:
                    f.write(str(idx) + ' ')
                    
        if enable_beat:
            with open(os.path.join(beated_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_beat:
                    f.write(str(idx) + ' ')            
    print('finish')


def build_files(num_pieces,
                min_length,
                lines=None, 
                finals=None,
                sentences=None,
                pos=None,
                beats=None,
                tokenized_data_path=None, 
                finalized_data_path=None,
                sentenced_data_path=None,
                posed_data_path=None,
                beated_data_path=None,
                full_tokenizer=None, 
                full_finalizer=None,
                full_sentencer=None,
                full_poser=None,
                full_beater=None,
                enable_final=False,
                enable_sentence=False,
                enable_pos=False,
                enable_beat=False,
                segment=False):
    print('Start tokenizing..')
    assert len(lines) == len(finals) == len(sentences)
    if segment:
        lines = segment_text(lines)
    path = tokenized_data_path.rsplit('/', 1)[0]
    if not os.path.exists(path):
        os.mkdir(path)
    print(f'#lines: {len(lines)}')
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path) 
    if enable_final:
        print(f'#finals: {len(finals)}')
        if not os.path.exists(finalized_data_path):
            os.mkdir(finalized_data_path)
    if enable_sentence:
        print(f'#sentences: {len(sentences)}')
        if not os.path.exists(sentenced_data_path):
            os.mkdir(sentenced_data_path)
    if enable_pos:
        print(f'#pos: {len(pos)}')
        if not os.path.exists(posed_data_path):
            os.mkdir(posed_data_path)
    if enable_beat:
        print(f'#beats: {len(beats)}')
        if not os.path.exists(beated_data_path):
            os.mkdir(beated_data_path)
    
    all_len = len(lines)
    for k in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
        if k == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece      
            
        if enable_final:
            subfinals = finals[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subfinals.extend(finals[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece
                     
        if enable_sentence:
            subsentences = sentences[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subsentences.extend(sentences[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece   
                
        if enable_pos:
            subpos = pos[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subpos.extend(pos[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece
        
        if enable_beat:
            subbeats = beats[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subbeats.extend(beats[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece
        
        for i in range(len(sublines)):
            line = sublines[i]
            if len(line) > min_length:
#                 print(len(line), line)
                line = full_tokenizer.tokenize(line)
#                 print(len(line), line)
                line = full_tokenizer.convert_tokens_to_ids(line)
                
                if enable_final:
                    final = subfinals[i]
                    final = full_finalizer.tokenize(final)
                    final = full_finalizer.convert_tokens_to_ids(final)  
                    assert len(final) == len(line)
                           
                if enable_sentence:
                    sentence = subsentences[i]
                    sentence = full_sentencer.tokenize(sentence)
                    sentence = full_sentencer.convert_tokens_to_ids(sentence)
                    assert len(sentence) == len(line)
                    
                if enable_pos:
                    p = subpos[i]
                    p = full_poser.tokenize(p)
                    p = full_poser.convert_tokens_to_ids(p)
                    assert len(p) == len(line)
                
                if enable_beat:
                    beat = subbeats[i]
                    beat = full_beater.tokenize(beat)
                    beat = full_beater.convert_tokens_to_ids(beat)
                    assert len(beat) == len(line)
                
                sublines[i] = line
                if enable_final:
                    subfinals[i] = final
                if enable_sentence:
                    subsentences[i] = sentence
                if enable_pos:
                    subpos[i] = p
                if enable_beat:
                    subbeats[i] = beat
        
        full_line, full_final, full_sentence, full_pos, full_beat = [], [], [], [], []
        for i in range(len(sublines)):
            mask = full_tokenizer.convert_tokens_to_ids('[MASK]')
            clss = full_tokenizer.convert_tokens_to_ids('[CLS]')
            full_line.append(mask)  # 文章开头添加MASK表示文章开始
            full_line.extend(sublines[i])
            full_line.append(clss)  # 文章之间添加CLS表示文章结束
            
            if enable_final:
                mask = full_finalizer.convert_tokens_to_ids('[MASK]')
                clss = full_finalizer.convert_tokens_to_ids('[CLS]')
                full_final.append(mask)  # 文章开头添加MASK表示文章开始
                full_final.extend(subfinals[i])
                full_final.append(clss)  # 文章之间添加CLS表示文章结束
            
            if enable_sentence:
                mask = full_sentencer.convert_tokens_to_ids('[MASK]')
                clss = full_sentencer.convert_tokens_to_ids('[CLS]')
                full_sentence.append(mask)  # 文章开头添加MASK表示文章开始
                full_sentence.extend(subsentences[i])
                full_sentence.append(clss)  # 文章之间添加CLS表示文章结束
                
            if enable_pos:
                mask = full_poser.convert_tokens_to_ids('[MASK]')
                clss = full_poser.convert_tokens_to_ids('[CLS]')
                full_pos.append(mask)  # 文章开头添加MASK表示文章开始
                full_pos.extend(subpos[i])
                full_pos.append(clss)  # 文章之间添加CLS表示文章结束    
            
            if enable_beat:
                mask = full_beater.convert_tokens_to_ids('[MASK]')
                clss = full_beater.convert_tokens_to_ids('[CLS]')
                full_beat.append(mask)  # 文章开头添加MASK表示文章开始
                full_beat.extend(subbeats[i])
                full_beat.append(clss)  # 文章之间添加CLS表示文章结束 
        
        if enable_final:
            assert len(full_line) == len(full_final), f'line: {len(full_line)}, final: {len(full_final)}'
        if enable_sentence:
            assert len(full_line) == len(full_sentence), f'line: {len(full_line)}, sentence: {len(full_sentence)}'
        if enable_pos:
            assert len(full_line) == len(full_pos), f'line: {len(full_line)}, pos: {len(full_pos)}'
        if enable_beat:
            assert len(full_line) == len(full_beat), f'line: {len(full_line)}, beat: {len(full_beat)}'
        
        with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
            for idx in full_line:
                f.write(str(idx) + ' ')
        
        if enable_final:
            with open(os.path.join(finalized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_final:
                    f.write(str(idx) + ' ')
        
        if enable_sentence:
            with open(os.path.join(sentenced_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_sentence:
                    f.write(str(idx) + ' ')
                    
        if enable_pos:
            with open(os.path.join(posed_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_pos:
                    f.write(str(idx) + ' ')
                    
        if enable_beat:
            with open(os.path.join(beated_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_beat:
                    f.write(str(idx) + ' ')           
    print('finish')
    

""" Processing Lyrics Data """
def process_lyric(ins_path='data/lyrics/RAP_DATASET_LYRIC/', out_path='data/lyrics/RAP_DATASET_LYRIC_valid/', invalid_songs=set([])):
    """
    处理歌词：去除非歌词部分（如歌名、作者等），删除空行，无意义行（例如注释演唱者）
    homepath = '/ssddata/lxueaa/controllable-text-generation/data'
    lyric_base = f'{homepath}/lyrics/RAP_DATASET_LYRIC'
    :return: list of invalid song path
    """
    i = 0  # total num
    j = 0  # number of empty songs

    # enumerate singers
    for rap_name in os.listdir(ins_path):
        rap_path = os.path.join(ins_path, rap_name)

        if os.path.isdir(rap_path):
            # enumerate album dirs
            for s_name in os.listdir(rap_path):
                s_path = os.path.join(rap_path, s_name)

                if os.path.isdir(s_path):
                    
                    # enumerate songs
                    for song_name in os.listdir(s_path):
                        i += 1
                        lyric_path = os.path.join(s_path, song_name, f'{song_name}_content.txt')
                        if os.path.exists(lyric_path):
                            finals_path = os.path.join(s_path, song_name, f'{song_name}_finals.txt')
                            with open(finals_path, 'w') as of:
                                with open(lyric_path) as f:
                                    for line in f:
                                        r = line.index(']')
                                        time = line[:r+1]
                                        content = line[r:]
                                        finals = get_sentence_pinyin_finals(content)
                                        finals = ' '.join(finals).rstrip(' \r\n')
                                        
                                        of.write(f'{time + finals}\n')
                        else:
                            j += 1
                            invalid_songs.add(lyric_path)
    print(f'End. Total songs:  {i}, invalid songs: {j}, left songs: {i - j}')

    return invalid_songs
       

def read_lyrics(root_path, reverse=False):
    
    
    out_path = os.path.join(root_path, 'train')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # check whether preprocessed cache exists or not
    lines = []
    finals = []
    sentences = []
    pos = []
    beats = []
    
    reverse_str = '_reverse' if reverse else ''
    out_content_path = f'{out_path}/content{reverse_str}.json'
    out_finals_path = f'{out_path}/finals{reverse_str}.json'
    out_sentences_path = f'{out_path}/sentences{reverse_str}.json'
    out_pos_path = f'{out_path}/pos{reverse_str}.json'
    out_beats_path = f'{out_path}/beats{reverse_str}.json'
    
    
    # read cached data
    if os.path.exists(out_content_path) and os.path.exists(out_sentences_path) and \
       os.path.exists(out_finals_path) and os.path.exists(out_pos_path) and \
       os.path.exists(out_beats_path):
        # load cached data
        with open(out_content_path, encoding='utf8') as ins:
            for l in ins:
                lines.append(l) 
        with open(out_finals_path, encoding='utf8') as ins:
            for l in ins:
                finals.append(l)            
        with open(out_sentences_path, encoding='utf8') as ins:
            for l in ins:
                sentences.append(l)
        with open(out_pos_path, encoding='utf8') as ins:
            for l in ins:
                pos.append(l)
        with open(out_beats_path, encoding='utf8') as ins:
            for l in ins:
                beats.append(l)
        return lines, finals, sentences, pos, beats
    
   
    # If not exists, to preprocess data
    # process new data
    print('Start to read processed lyrics from dataset....')   
    ins_path = os.path.join(root_path, 'lyrics.json')
    with open(ins_path, encoding='utf8') as ins:
        # enumerate each line in the file
        # each line is an article
        i = j = 0
        for l in ins:
            song = eval(json.loads(l))
#             print(type(song))
            if song['valid']:
                if not reverse:
                    lines.append(song['歌曲'])
                    finals.append(song['韵母'])
                    pos.append(song['相对位置'])
                    beats.append(song['鼓点'])
                else:
                    lines.append(song['歌曲反向'])
                    finals.append(song['韵母反向'])
                    pos.append(song['相对位置反向'])
                    beats.append(song['鼓点反向'])
                sentences.append(song['句子'])
                i += 1
            else:
#                 print(l)
                j += 1
        print(f'valid: {i}, invalid: {j}')
        
    with open(out_content_path, mode='w', encoding='utf8') as f:
        for line in lines:
            f.write(f'{line}\n')  
    with open(out_finals_path, mode='w', encoding='utf8') as f:
        for final in finals:
            f.write(f'{final}\n')
    with open(out_sentences_path, mode='w', encoding='utf8') as f:
        for sentence in sentences:
            f.write(f'{sentence}\n')  
    with open(out_pos_path, mode='w', encoding='utf8') as f:
        for p in pos:
            f.write(f'{p}\n')
    with open(out_beats_path, mode='w', encoding='utf8') as f:
        for beat in beats:
            f.write(f'{beat}\n')
    return lines, finals, sentences, pos, beats


def get_beat_token(cnt, line):
    lines = line.split()
    beat = ['0'] * len(lines)
    for idx, item in enumerate(lines):
        if item == '[BEAT]':
            cnt += 1
            beat[idx] = str(cnt)
    beat = ' '.join(beat) + ' '
    return cnt, beat


def get_inner_pos(line):
    lines = line.split()
    pos = ['0'] * len(lines)
    cnt = 0
    for idx, item in enumerate(lines):
        if item in special_tokens:
            pos[idx] = item
        else:
            pos[idx] = str(cnt)
            cnt += 1
    pos = ' '.join(pos) + ' '
    return pos


def parse_lyric(l_content_path, l_finals_path, with_beat=False, beat_mode=0):
    lyric = ''
    lyric_reverse = ''
    sentence = ''
    with open(l_content_path) as f:
        num_line = 0
        valid = False
        for line in f:
            # line format: [00:12.338]抱着沙发 睡眼昏花 凌乱头发
            if ']' in line:
                j = line.index(']')
                line = line[j + 1:]
                if beat_mode == 1 and num_line == 0:
                    tempo = line[:3]
                    line = line[3:]

            # ignore begin lines
            if ':' in line or '：' in line:
                continue
            
            if with_beat:
                line = line.strip(' \r\n').lstrip(' ')
                if beat_mode == 1:
                        line_reverse = '[BEAT]'.join(line[::-1].split(']TAEB['))
                        if num_line == 0:
                            line = tempo + line
                            line_reverse = tempo + line_reverse
                elif beat_mode == 2:
                        line_reverse = line[::-1]
                        line_reverse = '[S]'.join(line_reverse.split(']S['))
                        line_reverse = '[M]'.join(line_reverse.split(']M['))
                        line_reverse = '[F]'.join(line_reverse.split(']F['))           
                else:
                    line_reverse = '[BEAT]'.join(line[::-1].split(']TAEB['))
                
            else:  
                line = line.strip(' \r\n')
                line_reverse = line[::-1]
            line = re.sub('\s+', '[PAD]', line)
            line_reverse = re.sub('\s+', '[PAD]', line_reverse)
            assert len(line) == len(line_reverse)
            
            if len(line) == 0:  # end of block
                if len(lyric) > 0:  # not start of the file
                    continue
            else:
                line_reverse += '[SEP]'
                line += '[SEP]'
                nSEP = len(re.findall('\[SEP\]', line))
                nPAD = len(re.findall('\[PAD\]', line))
                
                if with_beat:
                    nBEAT = len(re.findall('\[BEAT\]', line))
                    if beat_mode != 0:
                        nSMF = len(re.findall('\[S\]', line)) + \
                               len(re.findall('\[M\]', line)) + \
                               len(re.findall('\[F\]', line))
                    else:
                        nSMF = 0
                    
                    nids = len(line) - 4 * (nSEP + nPAD) - 5 * nBEAT - 2 * nSMF
                else:
                    nids = len(line) - 4 * (nSEP + nPAD)
                ids = [str(num_line) for k in range(nids)]
                    
                sentence += ' '.join(ids) + ' '
                num_line += 1
                
            lyric += line
            lyric_reverse += line_reverse
    
    final = final_reverse = ''
    innerpos = innerpos_reverse = ''
    beat = beat_reverse = ''
    cnt = rcnt = 0
    with open(l_finals_path) as f:
        num_line = 0
        for line in f:
            # line format: [00:12.338]抱着沙发 睡眼昏花 凌乱头发
            if ']' in line:
                i = line.index(']')
                line = line[i + 1:]
                if beat_mode == 1 and num_line == 0:
                    tempo = line[:4]
                    line = line[4:]

            # ignore begin lines
            if ':' in line or '：' in line:
                continue
            
            if with_beat:
                line = remove_prefix(line.strip(' \r\n'), '[SEP] ')
                line = remove_prefix(line, '[PAD] ')
                line = remove_suffix(line, ' [PAD]')
                line = remove_suffix(line, '[PAD]')
                line = remove_suffix(line, ' [SEP]')
                line = re.sub('(\[SEP\])', '[PAD]', line)
                line = re.sub('(\[PAD\]\s)+', '[PAD] ', line)
                if line == '[PAD]':
                    continue
                line = ' '.join(line.split())
            else:    
                line = line.strip(' \r\n')
            line_reverse = ' '.join(line.split()[::-1])
            if beat_mode == 1 and num_line == 0:
                line = tempo + ' ' + line
                line_reverse = tempo + ' ' + line_reverse
            
            if len(line) == 0:  # end of block
                if len(final) > 0:  # not start of the file
                    continue
            else:
                line_reverse += ' [SEP] '
                line += ' [SEP] '
                num_line += 1
            if with_beat:
                cnt, lbeat = get_beat_token(cnt, line)
                rcnt, lbeat_reverse = get_beat_token(rcnt, line_reverse)
            lpos = get_inner_pos(line)
            lpos_reverse = get_inner_pos(line_reverse)
                
            final += line
            final_reverse += line_reverse
            if with_beat:
                beat += lbeat
                beat_reverse += lbeat_reverse
            innerpos += lpos
            innerpos_reverse += lpos_reverse
            
    
    lyric, final, sentence, innerpos = lyric.strip(' \n'), final.strip(' \n'), sentence.strip(' \n'), innerpos.strip(' \n')
    lyric_reverse, final_reverse, innerpos_reverse = lyric_reverse.strip(' \n'), final_reverse.strip(' \n'), innerpos_reverse.strip(' \n')
    if with_beat:
        beat, beat_reverse = beat.strip(' \n'), beat_reverse.strip(' \n')

    len_lyric = len(lyric) - \
                4 * (len(re.findall('\[SEP\]', lyric)) + len(re.findall('\[PAD\]', lyric))) - \
                5 * len(re.findall('\[BEAT\]', lyric)) - \
                2 * (len(re.findall('\[S\]', lyric)) + len(re.findall('\[M\]', lyric)) + len(re.findall('\[F\]', lyric)))
    len_final = len(final.split())
    len_sentence = len(sentence.split())
    try:
        assert len_lyric == len_final == len_sentence
    except:
        print(len_lyric, len_final, len_sentence)
        print(lyric)
        print(final)
        print(l_content_path)
        return
    
    if num_line > 4:
        valid = True
#     print(lyric, lyric_reverse, final, final_reverse, sentence, innerpos, innerpos_reverse, beat, beat_reverse, valid, num_line)
    return lyric, lyric_reverse, final, final_reverse, sentence, innerpos, innerpos_reverse, beat, beat_reverse, valid, num_line


def prepare_lyrics(ins_path, out_path, with_beat=False, beat_mode=0):
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    out_path = os.path.join(out_path, 'lyrics.json')
    if os.path.exists(out_path):
        while True:
            ins = input('Found cached files...Continue to overwrite? (Y/N)\n')
            if ins == 'Y':
                print('Start to reprocess raw data...')
                break
            elif ins == 'N':
                print('Use cached files.')
                return
            else:
                print('Invalid inputs.')
            
 
    with open(out_path, 'w', encoding='utf8') as outs:
        l_info = {}  # lyric info
        # enumerate singers
        i = 0  # total num
        j = 0  # number of empty songs
        max_num_lines = 0
        for s_path in os.listdir(ins_path):
            l_info['歌手'] = s_path
            s_path = os.path.join(ins_path, s_path)

            if os.path.isdir(s_path):
                # enumerate album
                for a_path in os.listdir(s_path):
                    l_info['专辑'] = a_path
#                     if a_path == '过山车31[Disc.1]':
#                         print(s_path)
                    a_path = os.path.join(s_path, a_path)

                    if os.path.isdir(a_path):
                        # enumerate songs
                        for l_path in os.listdir(a_path):
                            l_file_name = l_path
                            l_path = os.path.join(a_path, l_path)
                                                  
                            if os.path.isdir(l_path):
                                # enumerate lyric
                                for l_song in os.listdir(l_path):
                                    l_info['歌名'] = l_file_name  # remove '_content.txt' extension
                                    
                                    if with_beat:
                                        if beat_mode == 0:
                                            if l_song != 'mapped_final_with_beat.txt':
                                                continue
                                            l_content_path = os.path.join(l_path, 'lyric_with_beat.txt')
                                            l_finals_path = os.path.join(l_path, 'mapped_final_with_beat.txt')
                                        elif beat_mode == 1:
                                            if l_song != 'mapped_final_with_beat_global.txt':
                                                continue
                                            l_content_path = os.path.join(l_path, 'lyric_with_beat_global.txt')
                                            l_finals_path = os.path.join(l_path, 'mapped_final_with_beat_global.txt')
                                        elif beat_mode == 2:
                                            if l_song != 'mapped_final_with_beat_local.txt':
                                                continue
                                            l_content_path = os.path.join(l_path, 'lyric_with_beat_local.txt')
                                            l_finals_path = os.path.join(l_path, 'mapped_final_with_beat_local.txt')
                                    else:
#                                         if l_song[-5] != 't':
                                        if l_song != 'mapped_final_with_beat.txt':
                                            continue
#                                         l_content_path = os.path.join(l_path, l_file_name+'_content.txt')
#                                         l_finals_path = os.path.join(l_path, l_file_name+'_mapped_finals.txt')
                                        l_content_path = os.path.join(l_path, 'lyric_with_beat.txt')
                                        l_finals_path = os.path.join(l_path, 'mapped_final_with_beat.txt')
                                    if os.path.isfile(l_content_path):
                                        l_info['歌曲'], l_info['歌曲反向'], l_info['韵母'], \
                                        l_info['韵母反向'], l_info['句子'], l_info['相对位置'], \
                                        l_info['相对位置反向'], l_info['鼓点'], l_info['鼓点反向'], \
                                        l_info['valid'], num_lines = parse_lyric(l_content_path, l_finals_path, with_beat, beat_mode)
#                                         print(l_info)
                                        if max_num_lines < num_lines:
                                            max_num_lines = num_lines

                                    l_info_str = str(l_info)
                                    outs.write(f'{json.dumps(l_info_str, ensure_ascii=False)}\n')
                                    if not l_info['valid']:
                                        j += 1

                                    i += 1
                                    if i % 1000 == 0:
                                        print(f'Processed songs:{i}', end='\r', flush=True)
  
    print(f'End. Total songs:  {i}, invalid songs: {j}, left songs: {i - j}, max line in song: {max_num_lines}.')

    
if __name__ == '__main__':

    prepare_lyrics(ins_path='data/lyrics/lyrics_with_finals_large', 
                   out_path='data/lyrics/lyrics/lyrics', 
                   with_beat=False, 
                   beat_mode=0)
    read_lyrics(path='data/lyrics/lyrics', 
                out_content_path='data/lyrics/lyrics/train/content',
                out_finals_path='data/lyrics/lyrics/train/finals',
                out_sentences_path='data/lyrics/lyrics/train/sentences',
                out_pos_path='data/lyrics/lyrics/train/pos',
                out_beats_path='data/lyrics/lyrics/train/beats',
                reverse=True, 
                with_beat=False,
                beat_mode=0)