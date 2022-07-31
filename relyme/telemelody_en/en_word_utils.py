from en_config import Form

import spacy
from icecream import ic
from collections import Counter
from string import punctuation
from phonemizer import phonemize
from phonemizer.separator import Separator

SEP = "[sep]"
PUNC = [',', '.', '?']
AUX = ["ADP", "AUX", "CONJ", "CCONJ", "DET"]
KEY_topk = 3

def get_syllables(text):
    phn = phonemize(
        text,
        language='en-us',
        backend='festival',
        separator=Separator(phone='_', word=' ', syllable=' @@'),
        strip=True,
        preserve_punctuation=True,
        njobs=4
    )

    return phn

nlp = spacy.load("en_core_web_sm")
def get_tag(text):
    text = text.replace(SEP, '.')
    doc = nlp(text.lower())
    return doc

def get_keyword(text):
    def get_hotwords(text):
        result = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN']
        for token in get_tag(text):
            if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                result.append(token.text)

        return result

    output = get_hotwords(text)
    hashtags = [ x[0] for x in Counter(output).most_common(KEY_topk) ]
    return hashtags

def sents2words(syl_sents) -> list:
    tmp = []
    flag = True
    words = []
    sents = [ s for s in syl_sents.split() if s != SEP ]
    for idx, syl in enumerate(sents):
        if idx < len(sents)-2 and "@@" in sents[idx+1]:
            flag = True
        else:
            flag = False

        if flag:
            tmp.append(syl)
        else:
            tmp.append(syl)
            words.append(tmp)
            tmp = []
    
    return words

def get_in_word_pos(syl_sents) -> list:
    """
    Return a list of True or False indicates whether this letter is the first of a word
    syl_sents: "k_r_ey_z @@iy l_ih_t @@ax_l th_ih_ng" (crazy little thing)
    output: [ False, True, False, True, False ]
    """
    words = sents2words(syl_sents)

    letter_pos = []
    for word in words:
        letter_pos += [False] + (len(word)-1) * [True]

    return letter_pos

def get_aux(sents):
    docs = get_tag(sents)
    aux = [ token.text for token in docs if token.pos_ in AUX ]
    return aux    

def get_aux_mask(sents, syl_sents):
    """
    Return a list with True or False indicates whether the letter is auxiliary word or not

    e.g.
    sents: "crazy little thing called love [sep]"
    syl_sents: "k_r_ey_z @@iy l_ih_t @@ax_l th_ih_ng k_ao_l_d l_ah_v [sep]"
    aux_mask: [False, False, False, False, False, False, False]
    """
    words = sents2words(syl_sents)
    docs = get_tag(sents)

    aux_mask = []
    for word, token in zip(words, docs):
        # ic(token.pos_)
        aux_mask += len(word) * [ bool(token.pos_ in AUX) ]

    return aux_mask

def get_keyword_mask(sents, syl_sents):
    """
    k > 0: 词首; k = 0: 词中; k = -1: 非keyword
    """
    sents = sents.replace(SEP, '')
    keys = get_keyword(sents)
    words = sents2words(syl_sents)

    for idx, k in enumerate(keys):
        sents = sents.replace(k, str((idx % 9) + 1))
        # ic(sents)

    k = []
    tmp = [ int(s) if s.isdigit() else -1 for s in sents.split() ]
    for tag, word in zip(tmp, words):
        k += len(word) * [tag]

    key_mask = [k[0]] + \
                [ 
                    0 
                    if (k[i-1] == k[i] and k[i] > 0) 
                    else k[i] 
                    for i in range(1, len(k)) 
                ]

    return key_mask

def get_structure_mask(syl_sents_punc:list, _struct):
    """ 根据字数和结构把每个字附上它所在的结构段落
    Inputs:
        syl_sents_punc: ['k_r_ey_z', '@@iy', 'l_ih_t', '@@ax_l', 'th_ih_ng', 'k_ao_l_d', 'l_ah_v', ',']
        struct: ["v_1 v_2"
    """
    punct_idx = [i for i, s in enumerate(syl_sents_punc) if s in PUNC]
    
    st = 0
    strct_mask = []
    for i, ed in enumerate(punct_idx):
        strct_mask.append((_struct.split()[i], (ed-st)))
        
        st = ed+1
    
    return strct_mask

def get_sents_form_mask(syl_sents_punc:list):
    """ 根据标点符号判断这个句子的句型: 
        ?: 问句
        ,: 未完成句
        .: 稳定句
        
        syl_sents_punc: ['k_r_ey_z', '@@iy', 'l_ih_t', '@@ax_l', 'th_ih_ng', 'k_ao_l_d', 'l_ah_v', ',']
    """
    punct = [(i, s)for i, s in enumerate(syl_sents_punc) if s in PUNC]
    
    st = 0
    sents_form = []
    for ed, p in punct:
        form = Form.ascend if p in [','] else Form.descend
        sents_form += [form] * (ed-st)
        
        st = ed+1
    
    return sents_form

def main():
    # text = "crazy little thing called love\ncrazy little thing called love"
    # text = "crazy little thing called love [sep]"
    text = "crazy little thing called love,"
    # text = [line.strip() for line in text.split('\n') if line]
    # print(get_syllables(text.split()))

    syl = "k_r_ey_z @@iy l_ih_t @@ax_l th_ih_ng k_ao_l_d l_ah_v [sep]"
    # print(get_in_word_pos(syl))

    # print(get_aux_mask(text, syl))
    # print(get_keyword_mask(text, syl))
    
    # str = "v_1"
    # print(get_structure_mask(text, syl, str.split()))
    # txt = ['k_r_ey_z', '@@iy', 'l_ih_t', '@@ax_l', 'th_ih_ng', 'k_ao_l_d', 'l_ah_v', ',']
    # print(get_sents_form_mask(txt))
    
    # with open('data/en/test/lyric.txt', 'r') as r:
    #     lyrics = r.readlines()
    
    # with open('data/en/test/_lyric.txt', 'w') as w:
    #     sep_lyr = []
    #     for lyric in lyrics:
    #         tmp = lyric.replace('.', f" {SEP} ")
    #         tmp = tmp.replace(',', f" {SEP} ")
            
    #         sep_lyr.append(tmp)
        
    #     w.writelines(''.join(sep_lyr))
            
    with open('data/en/test/_lyric.txt', 'r') as r:
        lyrics = r.readlines() 
    
    syl = []
    for lyric in lyrics:
        sep_idx = [ i for i, s in enumerate(lyric.split()) if s == SEP ]
        sent = [ s for s in lyric.split() if s != SEP ]
        
        phn = get_syllables(sent)
        
        # print(phn)
        
        for i in sep_idx:
            phn.insert(i, SEP)
        
        syl.append(' '.join(phn))

    with open('data/en/test/_syllables.txt', 'w') as w:
        w.writelines('\n'.join(syl))
            
    # print(syl)

if __name__ == "__main__":
    main()