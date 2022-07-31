from config import interval_range, WEAK_BEAT, PitchPara, RhyPara

from icecream import ic
from pypinyin import lazy_pinyin, Style
from word_utils import get_aux_mask, get_in_word_pos
alpha, beta = 0.7, 0.3

SEP = '[sep]'
REST = 128
class Song():
    def __init__(self, lyric, melody) -> None:
        self.lyric = self.lyr_transform(lyric)
        self.tone = self.tone_transform()
        self.melody, self.rest = self.mld_transform(melody)
        self.pos = self.get_all_pos(melody)
        ly = ''.join(sum(self.lyric, []))
        ic(ly)
        self.aux_mask = self.aux_transform(get_aux_mask(ly))
        self.in_word_pos = self.in_word_transform(get_in_word_pos(ly))
        self.sent_cnt = 0

    def lyr_transform(self, _lyrics):
        tmp = []
        lyric = []
        # ic(_lyrics)
        for l in _lyrics:
            if l == SEP:
                tmp[-1] = f'{tmp[-1]}.'
                lyric.append(tmp)
                tmp = []
            else:
                tmp.append(l)
        
        return lyric 
    
    def mld_transform(self, _melody):
        i = 0
        tmp = []
        rest_tmp = []
        melody_sent = []
        rest = []
        while i < len(_melody):
            if _melody[i] != SEP:
                
                if int(_melody[i]) < REST:
                    tmp.append((int(_melody[i]), int(_melody[i+1])-REST))
                    
                if int(_melody[i]) == REST:
                    rest_tmp.append((int(_melody[i]), int(_melody[i+1])))
            
            else:
                melody_sent.append(tmp)
                rest.append(rest_tmp)
                tmp = []
                rest_tmp = []
  
            i += 1
           
        return melody_sent, rest
    
    def aux_transform(self, _aux_mask):
        i = 0
        aux = []
        for sent in self.lyric:
            
            aux.append(_aux_mask[i:i+len(sent)])
            i += len(sent)
            
        return aux
    
    def in_word_transform(self, _in_word):
        i = 0
        in_word = []
        for sent in self.lyric:
            in_word.append(_in_word[i:i+len(sent)])
            i += len(sent)
            
        return in_word
    
    def tone_transform(self):
        def get_tone_id(s) -> int:
            if s in ["1", "2", "3", "4"]:
                return int(s) - 1
            return 4
        
        tone = []
        for sent in self.lyric:
            # last one is punct
            _tone = lazy_pinyin(sent, style=Style.TONE3)[:-1]
            tone.append([ get_tone_id(t[-1]) for t in _tone ])
    
        return tone
        
    def get_all_pos(self, _melody):
        pos = []
        last_dur = 0
        i = 0 
        for note in _melody:
            if note == SEP: continue
            dur = int(note) - REST
            if dur > 0:
                pos.append(last_dur)
                last_dur += dur
                
        return pos
            
    def get_pos(self, note_idx):
        note_idx = min(note_idx, len(self.melody)-1)
        return self.pos[note_idx] % 4
    
    def get_note(self, note_idx):
        curr_sent = self.melody[self.sent_cnt]
        note_idx = min(note_idx, len(curr_sent)-1)
        return curr_sent[note_idx]
    
    def get_lyric(self, lyr_idx):
        curr_sent = self.lyric[self.sent_cnt]
        lyr_idx = min(lyr_idx, len(curr_sent)-1)
        return curr_sent[lyr_idx]
    
    def get_tone(self, lyr_idx):
        curr_sent = self.tone[self.sent_cnt]
        lyr_idx = min(lyr_idx, len(curr_sent)-1)
        return curr_sent[lyr_idx]
    
    def get_aux_flag(self, lyr_idx):
        curr_sent = self.tone[self.sent_cnt]
        
        lyr_idx = min(lyr_idx, len(self.lyric)-1)
        return self.aux_mask[lyr_idx]
    
    def print_debug(self):
        ic(self.lyric)
        ic(self.melody)
        ic(self.pos)
        ic(self.tone)
        ic(self.aux_mask)
        ic(self.in_word_pos)
        ic(self.rest)
        


def get_pitch_score(lyr_idx:int, mld_idx:int, song:Song):
    """get pitch score

    Args:
        lyr_idx (int): _description_
        mld_idx (int): _description_
        song (Song): _description_
    """
    def get_transition_score(delta_notes, prev_tone, curr_tone):
        note_r = interval_range[prev_tone][curr_tone]
        
        transition_score = PitchPara.P_WORST.value
        # Transitions
        if delta_notes >= note_r[0][0] and delta_notes <= note_r[0][1]:
            transition_score = PitchPara.P_BEST.value

        elif delta_notes >= note_r[1][0] and delta_notes <= note_r[1][1]:
            transition_score = PitchPara.P_SECO.value

        elif delta_notes >= note_r[2][0] and delta_notes <= note_r[2][1]:
            transition_score = PitchPara.P_THIR.value
    
        return transition_score
    
    prev_tone, curr_tone = song.get_tone(lyr_idx-1), song.get_tone(lyr_idx)
    
    # lyr-mld -> many-one
    if mld_idx < 0:
        mld_idx = -mld_idx
        # will be the same notes
        pitch_score = get_transition_score(0, prev_tone, curr_tone)
        
    # lyr-mld -> one-many or one-one
    else:
        prev_pitch, curr_pitch = song.get_note(mld_idx-1)[0], song.get_note(mld_idx)[0]
        pitch_score = get_transition_score(curr_pitch-prev_pitch, prev_tone, curr_tone)
    
    return pitch_score

def get_rhythm_score(lyr_idx:int, mld_idx:int, song:Song):
    """get rhythm score

    Args:
        lyr_idx (int): _description_
        mld_idx (int): _description_
        song (Song): _description_
    """
    def get_aux_score(aux_flag, pos):
        aux_score = RhyPara.BEST.value
        if pos in WEAK_BEAT and aux_flag:
            aux_score = RhyPara.WORST.value

        return aux_score

    pos = song.get_pos(mld_idx)
    aux_flag = song.get_aux_flag(lyr_idx)

    rhythm_score = get_aux_score(aux_flag, pos)
    return rhythm_score
    

# class dp_constraint():
#     def __init__(self, lyc, mld) -> None:
#         self.lyc = lyc
#         self.mld = mld
        
#         self.aux_mask = get_aux_mask(lyc)
#         self.key_mask = get_key_mask(lyc)
    
    
#     def get_pitch_score(self):
    
#     def get_cons_score(self, idx):
#         # try:
#         curr_note = mld[idx]
#         curr_lyrc = lyc[idx]
#         next_note = mld[idx+1] 
#         next_lyrc = lyc[idx+1]
#         # except IndexError:
#         #     return 
#         pitch_score = self.get_pitch_score(curr_note, curr_lyrc, next_note, next_lyrc)
#         rhythm_score = self.get_rhythm_score(curr_note, idx)
#         # structure_score = get_strucutre_score(curr_note, next_note)
        
#         score = alpha * pitch_score + beta * rhythm_score
        
#         return score
    
# print(lazy_pinyin(['你', '好.'], style=Style.TONE3))