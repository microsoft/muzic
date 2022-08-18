import random

def check(arr, m, a1, a2, mod1, mod2):
    n = len(arr)
    aL1, aL2 = pow(a1, m, mod1), pow(a2, m, mod2)
    h1, h2 = 0, 0
    for i in range(m):
        h1 = (h1 * a1 + arr[i]) % mod1
        h2 = (h2 * a2 + arr[i]) % mod2
    seen = dict()
    seen[(h1, h2)] = [m - 1] 
    for start in range(1, n - m + 1):
        h1 = (h1 * a1 - arr[start - 1] * aL1 + arr[start + m - 1]) % mod1
        h2 = (h2 * a2 - arr[start - 1] * aL2 + arr[start + m - 1]) % mod2
        if (h1, h2) in seen:
            if min(seen[(h1, h2)]) < start: 
                return start
            else:
                seen[(h1,h2)].append(start + m - 1)
        else:
            seen[(h1, h2)] = [start + m - 1]
        #seen.add((h1, h2))
    return -1

def longestDupSubstring(arr):

    a1, a2 = random.randint(26, 100), random.randint(26, 100)

    mod1, mod2 = random.randint(10**9+7, 2**31-1), random.randint(10**9+7, 2**31-1)

    n = len(arr)
    l, r = 1, n-1
    length, start = 0, -1

    while l <= r:
        m = l + (r - l + 1) // 2
        idx = check(arr, m, a1, a2, mod1, mod2)
        if idx != -1:
            l = m + 1
            length = m
            start = idx
        else:
            r = m - 1
    return start, length

def KMP_search(s,p,parent,init): 
    def buildNext(p):
        nxt = [0]
        x = 1
        now = 0
        while x < len(p):
            if p[now] == p[x]:
                now += 1
                x += 1
                nxt.append(now)
            elif now:
                now = nxt[now - 1]
            else:
                nxt.append(0)
                x += 1
        return nxt

    tar = 0
    pos = 0
    nxt = buildNext(p)
    is_first = True
    while tar < len(s):
        if s[tar] == p[pos] and (init or parent[tar] == -1):
            tar += 1
            pos += 1
        elif pos and (init or parent[tar] == -1):
            pos = nxt[pos - 1]
        else:
            tar += 1

        if pos == len(p):
            if is_first: # first matching
                is_first = False
                parent_start_idx = tar - pos
            else:
                parent[tar - pos:tar] = list(range(parent_start_idx,parent_start_idx+pos))
            pos = 0  # different from a standard kmp, here substrings are not allowed to overlap. So the pos is not nxt[pos - 1] but 0
    return parent

def Lyrics_match(sentence):
    """
    Recognition algorithm.
    First, we find (L,K) repeat which is like a longest repeated substring problem. A solution can be found in https://leetcode-cn.com/problems/longest-duplicate-substring/solution/zui-chang-zhong-fu-zi-chuan-by-leetcode-0i9rd/
    The code here refers to this solution.

    Then we use a modified KMP to find where does the first (L,K) repeat begins.
    """
    # sentence = lyrics.strip().split(' ')
    all_words = word_counter = [len(i) for i in sentence]
    parent = [-1] * len(word_counter)

    init = 0
    chorus_start = -1
    chorus_length = -1

    while True:
        start, length = longestDupSubstring(word_counter)
        if chorus_length >= len(parent) * 0.4 and init == 1:
            chorus_start = start
            chorus_length = length
            print(chorus_start, chorus_length)
            init += 1

        if init == 0:
            chorus_start = start
            chorus_length = length
            init += 1
        
        if start < 0 or length < 3:
            break

        p = word_counter[start:start + length]
        parent = KMP_search(all_words, p, parent, init)

        tmp = list()
        for i in range(len(word_counter)):
            if parent[i] == -1:
                tmp.append(word_counter[i])

        word_counter = tmp
        # start, length = longestDupSubstring(word_counter)
    # print('for test:',parent)
    # print('length:',len(parent))
    for idx in range(1, len(all_words)):
        if parent[idx] == -1 and all_words[idx - 1] == all_words[idx]:
            parent[idx] = -2
            if parent[idx - 1] == -2:
                parent[idx - 1] = idx - 2
        if parent[idx] >= 0 and parent[parent[idx]] != -1 and parent[parent[idx]] != -2:
            parent[idx] = parent[parent[idx]]

    parent[-1] = -1
    
    return parent, parent[chorus_start], chorus_length
    # return [-1] * len(parent), -1, -1 # ablation, when no structure
