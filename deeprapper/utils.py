#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from pypinyin import lazy_pinyin, Style
import torch
import torch.nn.functional as F
import math

valid_finals = ['a', 'o', 'e', 'i', 'u', 'v',
                    'ai', 'ei', 'ui', 'ao', 'ou', 
                    'iu', 'ie', 've', 'er', 'an', 
                    'en', 'in', 'un', 'vn', 
                    'ang', 'eng', 'ing', 'ong', 'n']

special_tokens = ['[SEP]', '[MASK]', '[PAD]', '[CLS]', '[UNK]', '[BEAT]', '[S]', '[M]', '[F]']

map_dict = {'a': 'a', 'o': 'o', 'e': 'e', 'i': 'i', 'v': 'i', 'u': 'u',
            'ai': 'ai', 'ei': 'ei', 'ui': 'ei', 'ao': 'ao',
            'ou': 'ou', 'iu': 'ou', 'ie': 'ie', 've': 'ie',
            'er': 'er', 'an': 'an', 'en': 'en', 'un': 'en',
            'in': 'in', 'vn': 'in', 'ang': 'ang', 'eng': 'en',
            'ing': 'in', 'ong': 'ong', 'n': 'en', 'UNK': 'UNK',
            '[SEP]': '[SEP]', '[PAD]': '[PAD]', '[BEAT]': '[BEAT]',
            '[F]': '[F]', '[M]': '[M]', '[S]': '[S]', '[UNK]': '[UNK]'}


def swap_value(arr, a, b):
    tmp = arr[a]
    arr[a] = arr[b]
    arr[b] = tmp

def logit2prob(logit):
    odds = math.exp(logit)
    prob = odds / (1.0 + odds)
    return prob

def temperature_softmax(logits, T=1):
    z = torch.exp(logits/T)
    return z / torch.sum(z, dim=-1)

def get_sentence_pinyin_finals(line, invalids_finals={}):
    
    finals = lazy_pinyin(line, style=Style.FINALS)
    valid = True
    for i in range(len(finals)):
        if finals[i] not in valid_finals:
            if 'invalid_1' not in invalids_finals:
                invalids = invalids_finals['invalid_1'] = {}
            else:
                invalids = invalids_finals['invalid_1']
                
            if finals[i] not in invalids:
#                 print(f'1, {line[i]}: {finals[i]}')
                invalids[finals[i]] = set([line[i]])
            else:
                invalids[finals[i]].add(line[i])
                
            finals[i] = finals[i][1:]
            
        if finals[i] not in valid_finals:
            if 'invalid_2' not in invalids_finals:
                invalids = invalids_finals['invalid_2'] = {}
            else:
                invalids = invalids_finals['invalid_2']
                
            if finals[i] not in invalids:
#                 print(f'2, {line[i]}: {finals[i]}')
                invalids[finals[i]] = set([line[i]])
            else:
                invalids[finals[i]].add(line[i])
                
            finals[i] = 'UNK'
        finals[i] = map_dict[finals[i]]
            
    return finals, valid


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False