#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import math
from datetime import datetime
from utils import get_sentence_pinyin_finals, special_tokens
from beam_search import sample_sequence, fast_sample_sequence, beam_search_decode_nctx, beam_search_decode


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
    if pattern == 'fast':
        sample_fn =  fast_sample_sequence
    elif pattern == 'sample':
        sample_fn =  sample_sequence
    elif pattern == 'beam':
        sample_fn = beam_search_decode if args.n_ctx >= args.length else beam_search_decode_nctx
    else:
        raise Exception(f'No such generate pattern: {pattern}')

    return sample_fn(model, context, pinyin_dict, args, device=device)
    
    
def main():
    parser = argparse.ArgumentParser()

    # inference parameters
    parser.add_argument('--device', default='7', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=512, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=4, type=int, required=False, help='生成几个样本')
    parser.add_argument('--n_ctx', default='512', type=int, required=False, help='生成设备')
    
    # 各种语料库
    parser.add_argument('--tokenizer_path', default='tokenizations/chinese_dicts.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--finalizer_path', default='tokenizations/finals.txt', type=str, required=False, help='选择韵母词库')
    parser.add_argument('--sentencer_path', default='tokenizations/sentences.txt', type=str, required=False, help='选择句子词库')
    parser.add_argument('--poser_path', default='tokenizations/sentences.txt', type=str, required=False, help='选择相对位置词库')
    parser.add_argument('--beater_path', default='tokenizations/beats.txt', type=str, required=False, help='选择鼓点词库')
    
    # log dirs
    parser.add_argument('--save_samples_dir', default='samples', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--samples_sign', default='', type=str, required=False, help="样本文件名标志，用于筛选样本")
    parser.add_argument('--model_dir', default='model/lyrics/model_epoch30', type=str, required=False, help='模型路径')
    
    # inference settings
    parser.add_argument('--prefix', default='大海', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--model_config', default='model/lyrics/model_epoch30/config.json', type=str, required=False, help='模型参数')
#     parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--pattern', default='sample', help='采用什么模式生成文本: sample, fast, beam')
    parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
    parser.add_argument('--enable_final', action='store_true', help='是否加入韵母embedding')
    parser.add_argument('--enable_sentence', action='store_true', help='是否加入sentence embedding')
    parser.add_argument('--enable_relative_pos', action='store_true', help='是否加入inner-sentence positional embedding', required=False)
    parser.add_argument('--enable_beat', action='store_true', help='是否加入beat embedding', required=False)
    parser.add_argument('--reverse', action='store_true', help='是否反向生成')
    parser.add_argument('--with_beat', action='store_true', help='是否加入beat信息')
    parser.add_argument('--beat_mode', default=0, type=int, help='beat控制模式：0.不控制；2.global；3.local', required=False)
    parser.add_argument('--tempo', default=1, type=int, required=False, help='歌词速度:0-local controll; 1-slow; 2-medium; 3-fast')
    
    # beam seach param
    parser.add_argument('--beam_width', default=2, type=int, required=False, help='每次选则结果数目（子节点个数）')
    parser.add_argument('--beam_samples_num', default=5, type=int, required=False, help='搜索树的路径上限，超过则会剪枝')
    parser.add_argument('--beam_sample_select_sg', default='sample', type=str, required=False,
                        help='剪枝算法。sample: 按照得分概率采样，sort: 选取得分最高的')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='logits惩罚生成温度')
    parser.add_argument('--beam_cut_temperature', default=10, type=float, required=False, help='beam剪枝概率温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='从最高的topk选，采样子节点使用')
    parser.add_argument('--topp', default=0, type=float, required=False, help='从超过该概率阈值的里面选')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='已成词的概率惩罚')
    parser.add_argument('--dynamic_rhyme', action='store_true', help='是否使用动态韵脚（此处不是指换韵脚，指的是从前面句子读）')
    parser.add_argument('--rhyme_sentence_num', default=2, type=int, required=False, help='韵脚控制观察的句子数目')
    parser.add_argument('--rhyme_count', default=2, type=int, required=False, help='押韵字数')
    parser.add_argument('--rhyme_bonus', default=5, type=int, required=False, help='韵脚词的概率 logits 奖励值')
    parser.add_argument('--rhyme_alpha', default=.5, type=float, required=False, help='韵脚词的概率奖励系数')
    parser.add_argument('--rhyme_prob_bound', default=0.6, type=float, required=False, help='是否要压前句韵的概率bound, 1.0=不变韵')
    
    
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

            for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                if is_word(item) and is_word(text[i + 1]):
                    text[i] = item + ' '
            for i, item in enumerate(text):
                if item == '[MASK]':
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
