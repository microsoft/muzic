# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
from tqdm import trange
import torch
import torch.nn.functional as F
from utils import get_sentence_pinyin_finals, top_k_top_p_filtering, special_tokens, temperature_softmax
import random


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, rhyme=[]):
        '''
        :param hiddenstate: current hidden state of GPT2
        :param previousNode: parent node
        :param wordId: input context at current state
        :param logProb: log probs of the sample sentence
        :param length: sentence length
        :param rhyme: rhyme words of later words
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.rhyme = rhyme

    def eval(self, alpha=1.0):
        base_score = math.exp(self.logp) #基础分，句子的概率：exp of sum of log p
        
        #TODO 可以在筛选过程中使用一些句子评分的规则作为额外的评分
        reward = 0

        return base_score + alpha * reward

    
def get_sentence(node):
    """
    Get the sentence from the leaf node
    """
    sen = []
    while node:
        words = node.wordid[0][0].tolist()
        for w in words[::-1]:
            sen.append(w)
        node = node.prevNode
    return sen[::-1]


def get_tokens(node, index):
    """
    Get the tokens from the leaf node
    Args:
    node: leaf node
    index: 0-token, 1-final, 2-sentence, 3-beat, 4-pos
    """
    sen = []
    while node:
        words = node.wordid[index][0].tolist()
        for w in words[::-1]:
            sen.append(w)
        node = node.prevNode
    return sen[::-1]


def _prepare_init_inputs(context, device):
    """
    build initial inputs for the model
    """
    context_tokens, context_finals, context_sentences, context_beats, context_poses = context
    
    context_tokens = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0) 
    
    if context_finals:
        context_finals = torch.tensor(context_finals, dtype=torch.long, device=device).unsqueeze(0) 
    
    if context_sentences:
        context_sentences = torch.tensor(context_sentences, dtype=torch.long, device=device).unsqueeze(0) 
    
    if context_beats:
        context_beats = torch.tensor(context_beats, dtype=torch.long, device=device).unsqueeze(0) 

    if context_poses:
        context_poses = torch.tensor(context_poses, dtype=torch.long, device=device).unsqueeze(0) 
    
    return context_tokens, context_finals, context_sentences, context_beats, context_poses
    

def _build_init_nodes(context, device):
    """
    Build initial inputs for beam search algo
    """    
    decoder_input = _prepare_init_inputs(context, device)
    root_node = BeamSearchNode(None, None, decoder_input, 0, len(context))
    
    return [root_node]


def _normalize_logits(logits, gen_tokens, tokenizer, temperature, repitition_penalty):
    """
    Normalize token logits: 降低已经生成词的概率, 降低 [UNK] 概率
    """
    for idx in set(gen_tokens): # 降低已经生成词概率
        logits[idx] /= repitition_penalty

    logits = logits / temperature
    logits[tokenizer.convert_tokens_to_ids('[UNK]')] = torch.min(logits) - 1  # unsafe: -float('Inf')
    
    return logits


def _sample_next_token_ids(logits, num_samples, top_k, top_p):
    """
    To sample most possible next tokens from predicted logits
    """
    # print('token logits shape: ', logits.shape)
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered_logits, dim=-1)
    next_token_ids = torch.multinomial(probs, num_samples=num_samples)
    
    return next_token_ids


def _generate_addtional_token_ids(next_token_id, gen_tokens, gen_sentences, gen_beats, gen_poses, 
                                  tokenizer, finalizer, sentencer, beater, poser, device):
    """
    generate additional token ids according to token id: final id, sentence id
    """
    next_token = tokenizer.convert_ids_to_tokens(next_token_id)[0]
    # get next final id
    if finalizer:
        if next_token in special_tokens:
            next_final = [next_token]
        else:
            next_final, _ = get_sentence_pinyin_finals(next_token) # next final shape: (1,)
        # shape: (1,1)
        next_final_id = torch.tensor(finalizer.convert_tokens_to_ids(next_final),
                                     dtype=torch.long, device=device).unsqueeze(0)
    else:
        next_final_id = None
     
    # get last token
    last_token_id = gen_tokens[0][-1]
    last_token = tokenizer.convert_ids_to_tokens([last_token_id])[0]

    # get next sentence id
    if sentencer:
        last_sentence_id = gen_sentences[0][-1]
        last_sentence = sentencer.convert_ids_to_tokens([last_sentence_id])[0]
        if next_token == '[MASK]' or next_token == '[CLS]':
            next_sentence = next_token
        elif last_token == '[SEP]':
            try:
                next_sentence = str(int(last_sentence) + 1)
            except:
                print(last_token, last_sentence)
        elif last_token == '[MASK]' or last_token == '[CLS]' :
            next_sentence = '0'
        else:
            next_sentence = last_sentence
        # shape: (1,1)
        next_sentence_id = torch.tensor(sentencer.convert_tokens_to_ids([next_sentence]),
                                        dtype=torch.long, device=device).unsqueeze(0)
    else:
        next_sentence_id = None
        
    
    # get next pos id
    if poser:
        if next_token in special_tokens:
            next_pos = next_token
        elif last_token == '[SEP]' or last_token == '[CLS]':
            next_pos = '0'
        else:
            for i in range(len(gen_poses[0])):
                last_pos_id = gen_poses[0][-i-1]
                last_pos = poser.convert_ids_to_tokens([last_pos_id])[0]
                if last_pos not in special_tokens: # search the most recently normal pos, skip special tokens
                    next_pos = str(int(last_pos) + 1)
                    break

        next_pos_id = torch.tensor(poser.convert_tokens_to_ids([next_pos]),
                                        dtype=torch.long, device=device).unsqueeze(0)
    else:
        next_pos_id = None
        
    
    # get beat id
    if beater: 
        if last_token == '[BEAT]':
            next_beat = '1' # default valud: the first beat
            
            # check previous beat
            for i in range(len(gen_beats[0])): 
                last_beat_id = gen_beats[0][-i-1]
                last_beat = beater.convert_ids_to_tokens([last_beat_id])[0]
                
                if last_beat not in ['0'] + special_tokens: # got the most recently beat
                    next_beat = str(int(last_beat) + 1)
                    break 
                # else skip non beat tokens
        else:
            next_beat = '0'
        next_beat_id = torch.tensor(beater.convert_tokens_to_ids([next_beat]),
                                        dtype=torch.long, device=device).unsqueeze(0)
    else:
        next_beat_id = None
    
    # shape: (1,1)
    return next_final_id, next_sentence_id, next_beat_id, next_pos_id


def _select_results(next_nodes, sample_select_sg, samples_num, temperature=10):
    if sample_select_sg == 'sample': # 按照概率采样
        probs = torch.tensor([n.eval() for n in next_nodes])
        next_ids = torch.multinomial(temperature_softmax(probs, temperature), num_samples=samples_num)
        nodes = []
        for ni in next_ids:
            nodes.append(next_nodes[ni])
    elif sample_select_sg == 'sort': # 保留概率最大的 samples_num 个样本
        next_nodes = sorted(next_nodes, key=lambda x: x.eval())
        nodes = next_nodes[:samples_num]
    else:
        raise Exception(f'No such sample_select_sg: {sample_select_sg}')
        
    return nodes


def _rescale_rhymes(probs, rhyme_word, tokenizer, beater, pinyin_dict, alpha=.5):
    
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    probs[sep_id] = 0 
    
    if beater:
        beat_id = tokenizer.convert_tokens_to_ids('[BEAT]')
        probs[beat_id] = alpha*probs[beat_id] + 1 - alpha
    
    rhyme_word = tokenizer.convert_ids_to_tokens(rhyme_word)
    rw_pinyin, valid = get_sentence_pinyin_finals(rhyme_word)
    if valid:
        rw_pinyin = rw_pinyin[0]
        probs[pinyin_dict[rw_pinyin]] = alpha*probs[pinyin_dict[rw_pinyin]] + 1 - alpha
    else:
        raise Exception(f'Invalid rhyme word: {rhyme_word}')
    
    return probs


def _control_rhymes(node, probs, tokenizer, beater, pinyin_dict, rhyme_words_list=None,
                    rhyme_count=2, sentence_num=2, rand_bound=0.5, alpha=.5):
    """
    control rhymes
    """
    # control rhymes
    if len(node.rhyme) > 0: # in rhyme process
        rhyme_words = node.rhyme[0]
        for w in rhyme_words:
            probs = _rescale_rhymes(probs, w, tokenizer, beater, pinyin_dict, alpha)
    else: # detect the begining of new sentences
        last_token_id = node.wordid[0][0][-1] #d2:batch size
        last_token = tokenizer.convert_ids_to_tokens([last_token_id])[0]
        
        if last_token == '[SEP]':
            if random.random() > rand_bound: # 概率不押之前的韵脚
                return probs
            
            # init rhymes
            if not rhyme_words_list:
                ss = get_sentence(node)
                ss = tokenizer.convert_ids_to_tokens(ss)
                ss = ''.join(ss).split('[SEP]')[-sentence_num-1:-1]
                rhyme_words_list = [[] for _ in range(rhyme_count)]
                for s in ss:
                    for spt in special_tokens:
                        s = s.replace(spt, '')
                    rhyme_words = tokenizer.convert_tokens_to_ids(list(s[:rhyme_count]))
                    for i, w in enumerate(rhyme_words):
                        rhyme_words_list[i].append(w) # 语句 s 中的第 i 个韵脚词
            
            # rescale rhymes
            # print('rhyme words list: ', rhyme_words_list)
            rhyme_words = rhyme_words_list[0]
            for w in rhyme_words:
                probs = _rescale_rhymes(probs, w, tokenizer, beater, pinyin_dict, alpha)
            
            node.rhyme = rhyme_words_list
            
    return probs


def beam_search_decode(model, context, pinyin_dict, args, device='cpu'):
    """
    Params:
        beam_width: 每次保留几个下一个节点
        sample_select_sg: 筛选样本策略。sample: 按照概率采样， sort: 选择概率最大的
        samples_num: 样本个数最大值
    
    Return:
        list of samples(ids)
    """
    
    nodes = _build_init_nodes(context[:5], device)
    tokenizer, finalizer, sentencer, beater, poser = context[-5:]
    if args.dynamic_rhyme:
        rhyme_words_list = None  
    else:
        rhymes_context = context[0]
        if args.beat_mode == 1:
            rhymes_context = rhymes_context[1:]
        rhyme_words_list = [[x] for x in rhymes_context]
    
    with torch.no_grad():
        for itr in trange(args.length):
            next_nodes = []
            for node in nodes:
                gen_tokens, gen_finals, gen_sentences, gen_beats, gen_poses = node.wordid
                
                # model predictions
                outputs = model(input_ids=gen_tokens, final_ids=gen_finals, sentence_ids=gen_sentences,
                                pos_ids=gen_poses, beat_ids=gen_beats, past_key_values=node.h)
                del node.h  # to release memory
                next_token_logits = _normalize_logits(outputs[0][0, -1, :], get_sentence(node), tokenizer,
                                                      args.temperature, args.repetition_penalty)
                token_probs = F.softmax(next_token_logits, dim=-1)
                
                # control rhymes
                token_probs =  _control_rhymes(node, token_probs, tokenizer, beater, pinyin_dict, rhyme_words_list,
                                               args.rhyme_count, args.rhyme_sentence_num, args.rhyme_prob_bound,
                                               args.rhyme_alpha)
                
                ## to get most possible next token ids
                probs_candidates, nti_candidates = torch.topk(token_probs, args.topk)
                next_token_ids = nti_candidates[torch.multinomial(probs_candidates, num_samples=args.beam_width)]

                # enumerate each possible token id
                for i in range(args.beam_width):

                    nt_id =  next_token_ids[i].unsqueeze(0)
                    log_p = torch.log(token_probs[nt_id[0]])
                    
                    beat_tokens = [get_tokens(node, 3)] if beater else None
                    pos_tokens =  [get_tokens(node, 4)] if poser else None
                    nf_id, ns_id, nb_id, np_id = _generate_addtional_token_ids(
                        nt_id, gen_tokens, gen_sentences, 
                        beat_tokens, pos_tokens, 
                        tokenizer, finalizer, sentencer,  
                        beater, poser, device
                    )
                     
                    # 生成[BEAT], 不占用韵脚词位置
                    next_token = tokenizer.convert_ids_to_tokens(nt_id)[0]
                    next_rhyme = node.rhyme if next_token == '[BEAT]' else node.rhyme[1:]
                
                    child_input = (nt_id.unsqueeze(0), nf_id, ns_id, nb_id, np_id) 
                    child_node = BeamSearchNode(outputs[1], node, child_input, node.logp + log_p, node.leng + 1, next_rhyme)
                    next_nodes.append(child_node)
                    
                    
            if len(next_nodes) <= args.beam_samples_num:
                nodes = next_nodes
            else: 
                nodes = _select_results(next_nodes, args.beam_sample_select_sg, args.beam_samples_num, args.beam_cut_temperature)
            
    results = [get_sentence(n) for n in nodes]
    return results

  
def beam_search_decode_nctx(model, context, length, n_ctx, tokenizer, finalizer, sentencer,  pinyin_dict,
                            beam_width=2, samples_num=5, sample_select_sg='sample', temperature=1.0, 
                            repitition_penalty=1.0, top_k=5, top_p=0.0, device='cpu', **kwargs):
    """
    Params:
        beam_width: 每次保留几个下一个节点
        sample_select_sg: 筛选样本策略。sample: 按照句子概率采样， sort: 选择句子概率最大的
        samples_num: 样本个数最大值
        n_ctx: 预测下一个词时，只用前 n_ctx 个词
        
    Return:
        list of samples(ids)
    """
    nodes = _build_init_nodes(context, device)
    
    with torch.no_grad():
        for itr in trange(length):
#             print(f'start iter: {itr}, samples num: {len(nodes) + len(results)}', end='\r', flush=True)
            next_nodes = []
            for node in nodes:
                generated_tokens, generated_finals, generated_sentences = node.wordid
                
                # to get most possible next token ids
                outputs = model(input_ids=generated_tokens[0][-(n_ctx - 1):].unsqueeze(0),
                                final_ids=generated_finals[0][-(n_ctx - 1):].unsqueeze(0),
                                sentence_ids=generated_sentences[0][-(n_ctx - 1):].unsqueeze(0))
                next_token_logits = _normalize_logits(outputs[0][0, -1, :], generated_tokens, tokenizer,
                                                      temperature, repitition_penalty)
                next_token_ids  = _sample_next_token_ids(next_token_logits, beam_width, top_k, top_p)
                
                # enumerate each possible token id
                token_probs = F.softmax(next_token_logits, dim=-1)
                for i in range(beam_width):

                    nt_id =  next_token_ids[i].unsqueeze(0)
                    log_p = math.log(token_probs[nt_id])
                    
                    nf_id, ns_id = _generate_addtional_token_ids(nt_id, generated_tokens, generated_sentences,
                                                                 tokenizer, finalizer, sentencer, device)

                    generated_tokens = torch.cat((generated_tokens, nt_id.unsqueeze(0)), dim=1)
                    generated_finals = torch.cat((generated_finals, nf_id.unsqueeze(0)), dim=1)
                    generated_sentences = torch.cat((generated_sentences, ns_id.unsqueeze(0)), dim=1)
                    child_input = (generated_tokens, generated_finals, generated_sentences)
                    child_node = BeamSearchNode(None, node, child_input, node.logp + log_p, node.leng + 1)
                    next_nodes.append(child_node)
                    
                    
            if len(next_nodes) <= samples_num:
                nodes = next_nodes
            else: 
                nodes = _select_results(next_nodes, sample_select_sg, samples_num)
                
    # get sentence            
    results = [n.wordid[0][0] for n in nodes]
    return results


def sample_sequence(model, context, length, n_ctx, tokenizer, finalizer, sentencer, pinyin_dict,
                    temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0, device='cpu'):
    """
    贪心策略：每次从概率 top_k 的下个词的候选中采样一个
    Args:
        n_ctx: 预测下一个词时，只用前 n_ctx 个词
    """
    
    generated_tokens, generated_finals, generated_sentences = _prepare_init_inputs(context, device)
    
    with torch.no_grad():
        for _ in trange(length):
            outputs = model(input_ids=generated_tokens[0][-(n_ctx - 1):].unsqueeze(0),
                            final_ids=generated_finals[0][-(n_ctx - 1):].unsqueeze(0),
                            sentence_ids=generated_sentences[0][-(n_ctx - 1):].unsqueeze(0))
            
            # to sample most possible next token id
            next_token_logits = _normalize_logits(outputs[0][0, -1, :], generated_tokens, tokenizer,
                                                      temperature, repitition_penalty)
            nt_id  = _sample_next_token_ids(next_token_logits, 1, top_k, top_p)
            nf_id, ns_id = _generate_addtional_token_ids(nt_id, generated_tokens, generated_sentences,
                                                         tokenizer, finalizer, sentencer, device)
            
            # @ code backup1
            
            # concatenate results
            generated_tokens = torch.cat((generated_tokens, nt_id.unsqueeze(0)), dim=1)
            generated_finals = torch.cat((generated_finals, nf_id.unsqueeze(0)), dim=1)
            generated_sentences = torch.cat((generated_sentences, ns_id.unsqueeze(0)), dim=1)
             
    return generated_tokens.tolist()


def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    """
    Can't be used at now!
    """
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return [generate]


# code backup 1
#             beat_logit = next_token_logits[tokenizer.convert_tokens_to_ids('[BEAT]')]
# #             print(f'old: {beat_logit}')
#             prob = logit2prob(beat_logit)
#             beat_prob = tempo * prob
# #             print(f'tempo={tempo}, prob={prob}, newp={beat_prob}')
#             beat_logit = math.log(beat_prob / (1.0 - beat_prob))  
# #             print(f'new: {beat_logit}')
#             next_token_logits[tokenizer.convert_tokens_to_ids('[BEAT]')] = beat_logit