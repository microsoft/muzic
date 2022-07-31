import numpy as np
from enum import Enum
from midi_utils import number_to_note

GEN_MODE = "BASE" # ReLyMe

TEMPO = 90

note_range = 1
first_note = 30
NOT_IN_KEY = [ first_note + interval + octave for octave in range(0, 60, 12) for interval in [0, 2, 4, 7, 9]]

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

class Sent(Enum):
    HEAD = 0 # 句首
    SECO = 1 # 句首后
    LAST = 2 # 句尾
    ELSE = 3 # 其他
    
class Form(Enum):
    ascend = 0   # 上行
    descend = 1  # 下行

class Strct(Enum):
    FIRST = 0 # 首句
    LAST = 1  # 末句
    ELSE = 3  # 其他

class PitchPara(Enum):
    TopK = 10
    Max_Same_Pitch = 4

    W_Strict = 1.5
    W_Middle = 1.2
    W_Looose = 1

    # Soft
    P_BEST = 0
    P_SECO = -0.5
    P_THIR = -1.5
    P_WORST = -4

    # bonus points for the sent form
    Sent_form_bias = 0.1

class PosPara(Enum):
    TopK = 20
    W = 1.5
    # Soft
    P_BEST = 0
    P_MIDD = -3
    P_WORST = -4
    
class PitchDebug():
    def __init__(
            self, 
            step,
            note_weight,
            curr_word, 
            prev_word, 
            prev_note, 
            beat, 
            sents_flag, 
            before_pr, 
            before_pi, 
            after_pr, 
            after_pi
            ):
        self.step = step
        self.note_weight = note_weight
        self.curr_word = curr_word
        self.prev_word = prev_word
        self.prev_note = number_to_note(prev_note)
        self.beat = beat
        self.sents_flag = sents_flag
        self.before_pr = before_pr
        self.before_pi = [ number_to_note(p) for p in before_pi ]
        self.after_pr = after_pr
        self.after_pi =  [ number_to_note(p) for p in after_pi ]
        
    def __repr__(self) -> str:
        print("Current Word: ", self.curr_word)
        print("Note Weight: ", self.note_weight)
        print("Previous Word: ", self.prev_word)
        print("Previous Note: ", self.prev_note)
        print("Beat: ", self.beat)
        print("Sents Flag: ", self.sents_flag)
        print(f"Before Modified:\n\tProbs:\n\t{self.before_pr}\n\tPitch:\n\t{self.before_pi}")
        print(f"AFTER  Modified:\n\tProbs:\n\t{self.after_pr}\n\tPitch:\n\t{self.after_pi}")

        return f"----- step: {self.step//4};{self.step} word: {self.curr_word} -----"

class PositionDebug():
    def __init__(
            self, 
            step,
            curr_word,
            prev_pos,
            prev_end,
            curr_bar,
            sents_pos,
            before_pr, 
            before_po, 
            after_pr, 
            after_po
            ):
        
        self.step = step
        self.curr_word = curr_word
        
        self.prev_bar = prev_pos // 16
        self.prev_pos = prev_pos % 16
        self.prev_end = prev_end % 16
        
        self.curr_bar = curr_bar
        
        self.sents_pos = sents_pos
        self.before_pr = before_pr
        self.before_po = before_po
        self.after_pr = after_pr
        self.after_po = after_po
        
    def __repr__(self) -> str:
        print("Current Bar: ", self.curr_bar)
        print("Previous: BAR {} ; ST {} ; ED {}".format(self.prev_bar, self.prev_pos, self.prev_end))
        # print("ATK Threshold: ", self.atk_thresho
        print("Sents Pos: ", self.sents_pos)
        print(f"Before Modified:\n\tProbs:\n\t{self.before_pr}\n\tPosition:\n\t{self.before_po}")
        print(f"AFTER  Modified:\n\tProbs:\n\t{self.after_pr}\n\tPosition:\n\t{self.after_po}")

        return f"----- step: {self.step//4};{self.step} word: {self.curr_word} -----"
