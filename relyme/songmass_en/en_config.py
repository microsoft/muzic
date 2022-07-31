import numpy as np
from enum import Enum
from midi_utils import number_to_note

GEN_MODE = "L2M" # BASE

TEMPO = 100

first_note = 30
NOT_IN_KEY = [ first_note + interval + octave for octave in range(0, 60, 12) for interval in [0, 2, 4, 7, 9]]
# print(NOT_IN_KEY)

RHY_DEBUG = True
POS_DEBUG = True
PIT_DEBUG = True

WEAK_BEAT = [1, 3]
TOKEN = ['Bar', 'Pos', 'Pitch', 'Dur']

np.set_printoptions(precision=4)
class ConstraintMode(Enum):
    N = 0     # Baseline
    P = 1     # Pitch Constraint only
    PB_1 = 2  # Beat Constraint but with only Aux and Keyword
    PB_2 = 3  # Full Constraints

class Form(Enum):
    ascend = 0   # 上行
    descend = 1  # 下行

class Strct(Enum):
    FIRST = 0 # 首句
    LAST = 1  # 末句
    ELSE = 3  # 其他

class PitchPara(Enum):
    TopK = 120
    Max_Same_Pitch = 4

    W_Strict = 1.5
    W_Middle = 1.5
    W_Looose = 1

    P_BEST = 50
    P_SECO = -0.5
    P_THIR = -1.5
    P_WORST = -20

    # bonus points for the sent form
    Sent_form_bias = 1.0

class PosPara(Enum):
    TopK = 20
    W = 1.5
    P_BEST = 20
    P_MIDD = -3
    P_WORST = -4

class PitchDebug():
    def __init__(
            self, 
            step,
            note_weight,
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
        self.prev_note = number_to_note(prev_note)
        self.beat = beat
        self.sents_flag = sents_flag
        self.before_pr = before_pr
        self.before_pi = [ number_to_note(p) for p in before_pi ]
        self.after_pr = after_pr
        self.after_pi =  [ number_to_note(p) for p in after_pi ]
        
    def __repr__(self) -> str:
        print("Note Weight: ", self.note_weight)
        print("Previous Note: ", self.prev_note)
        print("Beat: ", self.beat)
        print("Sents Flag: ", self.sents_flag)
        print(f"Before Modified:\n\tProbs:\n\t{self.before_pr}\n\tPitch:\n\t{self.before_pi}")
        print(f"AFTER  Modified:\n\tProbs:\n\t{self.after_pr}\n\tPitch:\n\t{self.after_pi}")

        return f"----- step: {self.step//4};{self.step} -----"

class PositionDebug():
    def __init__(
            self, 
            step,
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
        print("Sents Pos: ", self.sents_pos)
        print(f"Before Modified:\n\tProbs:\n\t{self.before_pr}\n\tPosition:\n\t{self.before_po}")
        print(f"AFTER  Modified:\n\tProbs:\n\t{self.after_pr}\n\tPosition:\n\t{self.after_po}")

        return f"----- step: {self.step//4};{self.step} -----"

def main():
    print(PitchPara.P_WORST.value)
    

if __name__ == "__main__":
    # print(POSPARA.W.value)
    # print(WEIGHT.STRICT.value)
    main()