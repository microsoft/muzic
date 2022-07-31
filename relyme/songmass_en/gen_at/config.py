import numpy as np
from enum import Enum

GEN_MODE = "L2M" # RERANK, L2M, BEATS, KEY, BASE

TEMPO = 90


POS_DEBUG = False
PIT_DEBUG = False

WEAK_BEAT = [1, 3]
np.set_printoptions(precision=4)

interval_range = [
    [[(0, 1), (-1, 2), (-3, 4)],  [(-5, -4), (-8, -3), (-10, -1)], [(-7, -6), (-12, -5), (-14, -2)], [(-4, -3), (-6, 1), (-8, 3)], [(-7, -6), (-12, -3), (-14, -1)]],
    [[(2, 3), (2, 5), (1, 7)],  [(-2, -1), (-3, 1), (-5, 3)], [(-4, -3),  (-6, 1), (-8, 3)], [(1, 2), (-1, 3), (-3, 6)], [(-4, -3), (-6, 1), (-8, 3)]],
    [[(4, 5), (3, 9), (1, 10)],  [(1, 2), (-1, 2), (-3, 5)], [(-4, -3), (-6, 1), (-8, 3)], [(3, 4), (0, 7), (-2, 8)], [(1, 2), (-1, 3), (-3, 5)]],
    [[(-1, 0), (-2, 2), (-4, 4)],  [(-4, -3), (-7, -1), (-9, 1)], [(-7, -6), (-12, -3), (-14, 0)], [(-6, -3), (-7, 1), (-12, 3)], [(-7, -6), (-12, -3), (-14, 0)]],
    [[(2, 3), (2, 5), (1, 7)],  [(1, 2), (-1, 3), (-3, 6)], [(-4, 3), (-6, 1), (-8, 3)], [(2, 3), (2,5), (1, 7)], [(2, 3), (2,5), (1, 7)]]
]


SONG_INDEX = ["倒带", "泡沫", "红豆", "红玫瑰", "达尔文", "青花瓷", "光年之外", "好久不见", "别找我麻烦", "缘分一道桥"]


class RhyPara(Enum):
    BEST = 1.0
    WORST = 0

class PitchPara(Enum):
    # Soft
    P_BEST = 5
    P_SECO = 3
    P_THIR = -1.5
    P_WORST = -4
    # Hard
    # P_BEST = 0
    # P_SECO = -15
    # P_THIR = -15
    # P_WORST = -15

    # bonus points for the sent form
    Sent_form_bias = 0.1

class PosPara(Enum):
    TopK = 20
    W = 1.5
    # Soft
    P_BEST = 0
    P_MIDD = -3
    P_WORST = -4
    
    # Hard
    # P_BEST = 0
    # P_MIDD = -15
    # P_WORST = -15 

def main():
    # for k in PitchPara:
    #     print(k.name, k.value)

    print(PitchPara.P_WORST.value)
if __name__ == "__main__":
    # print(POSPARA.W.value)
    # print(WEIGHT.STRICT.value)
    main()