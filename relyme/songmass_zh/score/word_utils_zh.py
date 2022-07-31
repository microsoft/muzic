import re
import pkuseg
import string
from pypinyin import lazy_pinyin, Style
from icecream import ic

import config
from _keyword import get_keyword

PUNC = [',', '.', ' ', '?']
AUX  = ['c', 'u', 'p'] 
TOPK = 3
KEY_FLAG = ('ns', 'n', 'vn', 'v', 'r', 'a', 'ad', 'd')

def split_sents(sents):
    """
    split sents by punctuation
    "aa, bbb, cc" -> ["aa", "bb", "cc"]
    """
    space = '$'

    if sents[-1] in PUNC:
        sents = sents[:-1]

    return sents.translate(str.maketrans({',': space, '.': space, ' ': ''})).split(space)

def get_sents_pos(sents) -> list:
    """
    "东方的太.阳." ->
    [('东', Sent.HEAD), ('方', Strct.FIRST), ('的', Strct.FIRST),
    ('太', Strct.FIRST), ('阳', Sent.HEAD)]
    To show the position info of each word, refer to config.py for meaning of each flag
    """
    global PUNC    
    _sents = []
    punc_cnt = 0
    num_punc = len([i for i in sents if i in PUNC])
    for i in range(len(sents)-1):
        if sents[i] in PUNC:
            punc_cnt += 1
            continue

        if sents[i-1] in PUNC:
            flag = config.Sent.HEAD
        elif punc_cnt == 0:
            flag = config.Strct.FIRST
        elif punc_cnt == num_punc - 1:
            flag = config.Strct.LAST
        elif sents[i+1] in PUNC:
            flag = config.Sent.LAST
        else:
            flag = config.Strct.ELSE

        _sents.append((sents[i], flag))

    return _sents

def get_letter_pos(sents) -> list:
    """
    返回每个letter在句子中的位置, for 对齐语句、乐句
    0: 句首; 1: 句首后; 2: 句尾; 3: else;
    Priority: 0 > 2 > 1 > 3
    """

    flag = config.Sent.HEAD
    letter_pos = []
    for idx, s in enumerate(sents):
        if s == '.':
            flag = config.Sent.HEAD
            continue

        if flag == config.Sent.HEAD:
            # 句首
            letter_pos.append(config.Sent.HEAD)
            flag = config.Sent.SECO

        elif idx < len(sents)-1 and sents[idx+1] == '.':
            # 句尾
            letter_pos.append(config.Sent.LAST)
            flag = config.Sent.HEAD

        elif flag == config.Sent.SECO:
            # 句首后
            letter_pos.append(config.Sent.SECO)
            flag = config.Sent.ELSE

        else:
            # else
            letter_pos.append(config.Sent.ELSE)        

    return letter_pos

def get_sents_form(sents):
    """ 根据标点符号判断这个句子的句型: 
        ?: 问句
        ,: 未完成句
        .: 稳定句
    """
    punct = []
    last_punc_idx = -1
    for idx, letter in enumerate(sents):
        if letter in ['?', ',']:
            punct.append((config.Form.ascend, idx-last_punc_idx-1))
            last_punc_idx = idx

        if letter in ['.']:
            punct.append((config.Form.descend, idx-last_punc_idx-1))
            last_punc_idx = idx

    sents_form = [ form for form, word_cnt in punct for _ in range(word_cnt) ]
    return sents_form

def get_structure(sents, _struct):
    """
    Inputs:
        sents: "你栽出千萬花的一生,四季中徑自盛放也凋零"
        struct: ["v_1", "v_2"]
    Outputs: [('v_1', 9), ('v_2', 10)]
    """
    ic(sents, _struct)
    strct = []
    st_id = 0
    last_punc_idx = -1
    for idx, letter in enumerate(sents):
        if letter in PUNC:
            strct.append((_struct[st_id], idx-last_punc_idx-1))
            st_id += 1
            last_punc_idx = idx

    return strct

def get_in_word_pos(sents) -> list:
    """
    Return a list of True or False indicates whether this letter is the first of a word
    sents: "你好吗"
    output: [ False, True, True ]
    """

    seg = pkuseg.pkuseg()
    sw_ = split_sents(''.join(sents))
    words = sum([ seg.cut(s) for s in sw_ ], [])

    # print(words)
    letter_pos = []
    for word in words:
        letter_pos += [False] + (len(word)-1) * [True]

    return letter_pos

def get_aux_mask(sents):
    """
    Return a list with True or False indicates whether the letter is auxiliary word or not

    e.g.
    sents: "金灿灿的太阳象父亲的目光."
    aux_mask: [False, False, False, True, False, False, False, False, False]
    """
    seg = pkuseg.pkuseg(postag=True)
    sw_ = split_sents(''.join(sents))
    words = [ seg.cut(s) for s in sw_ ]

    aux_mask = []
    for word in words:
        for w_, flag in word:
            aux_mask += len(w_) * [ bool(flag in AUX) ]

    return aux_mask

def get_keyword_mask(sents):
    """
    k > 0: First letter of the keyword; k = 0: Other letters of the keyword; k = -1: Not keyword
    """
    # remove punctuation
    sents = sents.translate(str.maketrans('', '', string.punctuation))
    keys = get_keyword(sents)

    key_mask = []
    for idx, k in enumerate(keys):
        sents = sents.replace(k, str((idx % 9) + 1) * len(k))

    tmp = [ int(s) if s.isdigit() else -1 for s in sents ]
    key_mask = [ tmp[0] ] + [ 0 if (tmp[i-1] == tmp[i] and tmp[i] > 0) else tmp[i] for i in range(1, len(tmp)) ]

    return key_mask

def clean_beat(lyrics, beat):
    """ Perform Matched Strong/Weak positions constraints

    Args:
        aux_mask (list): whether this letter is auxiliary word or not
        beat (list): original beat of this letter
        demo (str): the demo midi to be generated, use this para to find its keyword file

    Returns:
        clbeat (list): modified beat
    """
    cl_beat = beat.copy()
    # AUX Constraint
    aux_cnt = 0
    aux_mask = get_aux_mask(lyrics)
    for idx, (a, b) in enumerate(zip(aux_mask, beat)):
        if a and (b not in config.WEAK_BEAT):
            try: 
                prev_b = beat[idx-1]
                post_b = beat[idx+1]

                if prev_b <= b + 1 <= post_b:
                    cl_beat[idx] += 1
                    aux_cnt += 1

                elif prev_b <= b - 1 <= post_b:
                    cl_beat[idx] -= 1
                    aux_cnt += 1

            except IndexError:
                continue

    # Keyword Constraint
    sents = lyrics.translate(str.maketrans('', '', string.punctuation))
    key_mask = get_keyword_mask(sents)

    key_cnt = 0
    for idx, (a, b) in enumerate(zip(key_mask, beat)):
        if a > 0 and (b in config.WEAK_BEAT):
            try: 
                prev_b = beat[idx-1]
                post_b = beat[idx+1]

                if prev_b <= b - 1 <= post_b:
                    cl_beat[idx] -= 1
                    key_cnt += 1

                elif prev_b <= b + 1 <= post_b:
                    cl_beat[idx] += 1
                    key_cnt += 1

            except IndexError:
                continue

    ic(aux_cnt, key_cnt)
    return cl_beat

def clean(word):
    """
    clean up the sents
    """
    word = re.sub('[ \xa0]+', '', word)
    word = re.sub('[,，] *', ',', word)
    word = re.sub('\.{6} *', '.', word)
    word = re.sub('\…+ *', '.', word)
    word = word.strip('\n')

    if config.GEN_MODE == "BASE":
        word = re.sub('[。！？\?] *', '.', word)
    else:
        word = re.sub('[。！？] *', '.', word)

    return word


def main():
    sents = "你栽出千萬花的一生,四季中徑自盛放也凋零."
    struct = ["v_1", "v_2"]
    print(get_structure(sents, struct))


if __name__ == "__main__":
    main()
    