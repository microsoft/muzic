# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import math
from datetime import datetime
from utils import get_sentence_pinyin_finals, special_tokens
from beam_search import sample_sequence, beam_search_decode_nctx, beam_search_decode


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def _init_pinyin_dict(tokenizer):
    print('Initilizing pinyin dict')
    pinyin_dict = {}
    for i in range(tokenizer.vocab_size):
        w =  tokenizer.convert_ids_to_tokens(i)
        pinyin, valid = get_sentence_pinyin_finals(w)
        if valid:
            pinyin = pinyin[0]
            if pinyin in pinyin_dict:
                pinyin_dict[pinyin].append(i)
            else:
                pinyin_dict[pinyin] = [i]
    
    # display pinyin information
    print(f'Pinyin num: {len(pinyin_dict)}')
    kv_info = ''
    for k, v in pinyin_dict.items():
        kv_info += f'{k}:{len(v)}, '
    print(kv_info)
    # print(tokenizer.convert_ids_to_tokens(pinyin_dict['UNK']))
    return pinyin_dict


def generate(model, context, pinyin_dict, args, device='cpu'):
    
    pattern = args.pattern
    if pattern == 'sample':
        sample_fn =  sample_sequence
    elif pattern == 'beam':
        sample_fn = beam_search_decode if args.n_ctx >= args.length else beam_search_decode_nctx
    else:
        raise Exception(f'No such generate pattern: {pattern}')

    return sample_fn(model, context, pinyin_dict, args, device=device)
    
    
def main():
    parser = argparse.ArgumentParser()

    # inference parameters
    parser.add_argument('--device', default='7', type=str, required=False, help='cpu or gpu number')
    parser.add_argument('--length', default=512, type=int, required=False, help='sequence length')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='batch size')
    parser.add_argument('--nsamples', default=4, type=int, required=False, help='number of samples')
    parser.add_argument('--n_ctx', default='512', type=int, required=False, help='window of context')
    
    # path of dictionary
    parser.add_argument('--tokenizer_path', default='tokenizations/chinese_dicts.txt', type=str, required=False, help='vocabulary of tokens')
    parser.add_argument('--finalizer_path', default='tokenizations/finals.txt', type=str, required=False, help='vocabulary of finals')
    parser.add_argument('--sentencer_path', default='tokenizations/sentences.txt', type=str, required=False, help='vocabulary of sentences')
    parser.add_argument('--poser_path', default='tokenizations/sentences.txt', type=str, required=False, help='vocabulary of intra-sentence positions')
    parser.add_argument('--beater_path', default='tokenizations/beats.txt', type=str, required=False, help='vocabulary of beats')
    
    # log dirs
    parser.add_argument('--save_samples_dir', default='samples', type=str, required=False, help="path to save generated samples")
    parser.add_argument('--samples_sign', default='', type=str, required=False, help="name of samples")
    parser.add_argument('--model_dir', default='model/lyrics/model_epoch30', type=str, required=False, help='path to load models')
    
    # inference settings
    parser.add_argument('--prefix', default='大海', type=str, required=False, help='prefix given to the model')
    parser.add_argument('--model_config', default='model/lyrics/model_epoch30/config.json', type=str, required=False, help='model parameters')
    parser.add_argument('--segment', action='store_true', help='whehter to do Chinese Word Segmentation.')
    parser.add_argument('--pattern', default='sample', help='sample mode: beam')
    parser.add_argument('--save_samples', action='store_true', help='whether to save samples')
    parser.add_argument('--enable_final', action='store_true', help='whether to use final embedding')
    parser.add_argument('--enable_sentence', action='store_true', help='whether to use sentence embedding')
    parser.add_argument('--enable_relative_pos', action='store_true', help='whether to use intra-sentence positional embedding', required=False)
    parser.add_argument('--enable_beat', action='store_true', help='whether to use beat embedding', required=False)
    parser.add_argument('--reverse', action='store_true', help='whether to use reverse language model')
    parser.add_argument('--with_beat', action='store_true', help='whether to generate beats')
    parser.add_argument('--beat_mode', default=0, type=int, help='beat mode：0.no control；2.global；3.local', required=False)
    parser.add_argument('--tempo', default=1, type=int, required=False, help='pace of beats:0-local controll; 1-slow; 2-medium; 3-fast')
    
    # beam seach param
    parser.add_argument('--beam_width', default=2, type=int, required=False, help='beam width')
    parser.add_argument('--beam_samples_num', default=5, type=int, required=False, help='beam searching samples')
    parser.add_argument('--beam_sample_select_sg', default='sample', type=str, required=False,
                        help='sampleing algorithm. sample: sample according with scores，sort: choose the sample with highest scores')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='sampling temperature')
    parser.add_argument('--beam_cut_temperature', default=10, type=float, required=False, help='beam cut temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='sample from topk tokens')
    parser.add_argument('--topp', default=0, type=float, required=False, help='sample from topp tokens')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='repetition penalty')
    parser.add_argument('--dynamic_rhyme', action='store_true', help='whether to use dynamic rhyme（')
    parser.add_argument('--rhyme_sentence_num', default=2, type=int, required=False, help='checking rhyming according to the previous n sentences.')
    parser.add_argument('--rhyme_count', default=2, type=int, required=False, help='number of words rhyming')
    parser.add_argument('--rhyme_bonus', default=5, type=int, required=False, help='logits bonus given to rhyming words.')
    parser.add_argument('--rhyme_alpha', default=.5, type=float, required=False, help='probability bonus given to rhyming words.')
    parser.add_argument('--rhyme_prob_bound', default=0.6, type=float, required=False, help='probability of whether to use a new rhyme')
    
    
    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    
    ########################################################    
    # basic settings
    ###################################
    # set envs and import related packages
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    import torch
    import torch.nn.functional as F
    from module import GPT2LMHeadModel
    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert
    if args.device == 'cpu':
        device = 'cpu'
    else:
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ################################################################
    # load trained model
    #################
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.eval()
    model.to(device)
    
    
    ################################################################
    # init log writer
    #################
    if args.save_samples:
        save_samples_path = os.path.join(args.save_samples_dir, *args.model_dir.split('/')[1:], args.prefix)                
        if not os.path.exists(save_samples_path):
            os.makedirs(save_samples_path)
        samples_file = open(os.path.join(save_samples_path, f'{args.samples_sign}_samples_{datetime.now()}.txt'), 'w', encoding='utf8')    
    
    
    ################################################################
    # Prepare context
    #################
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path, do_lower_case=False)
    raw_text = args.prefix
    
    # to control beat frequency
    if args.beat_mode == 1 and args.tempo:
        tempos = ['[S]', '[M]', '[F]']
        tempo = [tempos[args.tempo-1]]
    else:
        tempo = []
        
    if args.reverse:
        special_token = []
        raw_text = raw_text[::-1] 
    context_tokens = tempo[0] + raw_text if tempo else raw_text
    context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context_tokens))
    
    
    if args.enable_final:
        finalizer = tokenization_bert.BertTokenizer(vocab_file=args.finalizer_path, tokenize_chinese_chars=False, do_lower_case=False)
    
        context_finals, _ = get_sentence_pinyin_finals(raw_text)
        context_finals = tempo + context_finals
        context_finals = finalizer.convert_tokens_to_ids(context_finals)
    else:
        context_finals = None
        finalizer = None
    
    if args.enable_sentence:
        sentencer = tokenization_bert.BertTokenizer(vocab_file=args.sentencer_path, tokenize_chinese_chars=False, do_lower_case=False)
        
        context_sentences = tempo + ['0'] * len(raw_text)
        context_sentences = sentencer.convert_tokens_to_ids(context_sentences)
    else:
        sentencer = None
        context_sentences = None
    
    if args.enable_beat:
        beater = tokenization_bert.BertTokenizer(vocab_file=args.beater_path, tokenize_chinese_chars=False, do_lower_case=False)
        
        context_beats = tempo + ['0']  * len(raw_text)
        context_beats = beater.convert_tokens_to_ids(context_beats)
    else:
        beater = None
        context_beats = None
        
    if args.enable_relative_pos:
        poser = tokenization_bert.BertTokenizer(vocab_file=args.poser_path, tokenize_chinese_chars=False, do_lower_case=False)
        context_poses = tempo + [str(i) for i in range(len(raw_text))]
        context_poses = poser.convert_tokens_to_ids(context_poses)
    else:
        poser = None
        context_poses = None

    context = [context_tokens, context_finals, context_sentences, context_beats, context_poses,
              tokenizer, finalizer, sentencer, beater, poser]
    
    # print('context:', context)
    #############################################################
    # Start to generate samples
    #####################################
    pinyin_dict = _init_pinyin_dict(tokenizer)
    generated = 0
    for _ in range(args.nsamples):
        outs = generate(model=model, context=context, pinyin_dict=pinyin_dict, args=args, device=device)
        
        
        # To display and save samples
        for out in outs:
            generated += 1
            # convert id to text tokens
            text = tokenizer.convert_ids_to_tokens(out)
            
            # To relace some special tokens
            if args.reverse:
                text = ''.join(text)
                for token in special_tokens:
                    if token == '[SEP]':
                        continue
                    text = token[::-1].join(text.split(token))
                text = text.split('[SEP]')    
                for i, piece in enumerate(text):
                    text[i] = text[i][::-1]
                text = '[SEP]'.join(text)
                text = tokenizer.tokenize(text)

            for i, item in enumerate(text[:-1]):  # ensuring space before english words.
                if is_word(item) and is_word(text[i + 1]):
                    text[i] = item + ' '
            for i, item in enumerate(text):
                if item == '[MASK]' or item == '[SKIP]':
                    text[i] = ''
                elif item == '[CLS]':
                    text[i] = '\n\n'
                elif item == '[SEP]':
                    text[i] = '\n'
                elif item == '[PAD]':
                    text[i] = ' '
            
            # print samples
            info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
            text = ''.join(text).replace('##', '').strip()
            print(info + text)
            
            # save samples
            if args.save_samples:
                samples_file.write(info + text + '\n' + '=' * 90 + '\n' * 2)
                samples_file.flush()
                
    print("=" * 80)
    # close file when finish writing.
    if args.save_samples:
        samples_file.close()


if __name__ == '__main__':
    main()
