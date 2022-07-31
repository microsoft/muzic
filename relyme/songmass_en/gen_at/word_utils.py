import pkuseg

AUX  = ['c', 'u', 'p'] # 语助词的词性
PUNC = [',', '.', '?']

def split_sents(sents):
    """
    split sents by punctuation
    "aa, bbb, cc" -> ["aa", "bb", "cc"]
    """
    space = '$'

    if sents[-1] in PUNC:
        sents = sents[:-1]

    return sents.translate(str.maketrans({',': space, '.': space, ' ': ''})).split(space)

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

# sents = "东方的太.阳.升在大海上.东方.的月亮挂在.蓝天上."
# print(len(get_in_word_pos(sents)))