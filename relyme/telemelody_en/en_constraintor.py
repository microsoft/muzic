import en_config as _config
from en_word_utils import get_aux_mask, get_keyword_mask

import re
import torch
from icecream import ic

# DEBUG = True
PUNC = [",", "."]
TOKEN = ['Bar', 'Pos', 'Pitch', 'Dur']


class BeatConstraintor():
    def __init__(self, lyrics, syllables, beat) -> None:
        self.lyrics = lyrics
        self.syllables = syllables
        self.beat = beat
        
        self.aux_mask = get_aux_mask(lyrics, syllables)
        self.key_mask = get_keyword_mask(lyrics, syllables)
        self.aux_debug = []
        self.key_debug = []
        
    def aux_constraint(self, beat=None):
        ori_beat = self.beat if beat is None else beat
        cln_beat = ori_beat.copy()
        
        aux_debug = []
        for idx, (a, b) in enumerate(zip(self.aux_mask, ori_beat)):
            if a and (b not in _config.WEAK_BEAT):
                try: 
                    prev_b = beat[idx-1]
                    post_b = beat[idx+1]

                    if prev_b <= b + 1 <= post_b:
                        cln_beat[idx] += 1
                        aux_debug.append((idx, b, cln_beat[idx]))

                    elif prev_b <= b - 1 <= post_b:
                        cln_beat[idx] -= 1
                        aux_debug.append((idx, b, cln_beat[idx]))
                
                except IndexError:
                    continue
    
        self.aux_debug = aux_debug
        return cln_beat
    
    def key_constraint(self, beat=None):
        ori_beat = self.beat if beat is None else beat
        cln_beat = ori_beat.copy()
        
        key_debug = []
        # ic(self.key_mask)
        for idx, (a, b) in enumerate(zip(self.key_mask, ori_beat)):
            if a > 0 and (b in _config.WEAK_BEAT):
                try: 
                    prev_b = beat[idx-1]
                    post_b = beat[idx+1]

                    if prev_b <= b - 1 <= post_b:
                        cln_beat[idx] -= 1
                        key_debug.append((idx, b, cln_beat[idx]))

                    elif prev_b <= b + 1 <= post_b:
                        cln_beat[idx] += 1
                        key_debug.append((idx, b, cln_beat[idx]))
        
                except IndexError:
                    continue
        
        self.key_debug = key_debug
        return cln_beat
    
    def rhythm_constraint(self):
        aux_beat = self.aux_constraint(self.beat)
        aux_key_beat = self.key_constraint(aux_beat)
        
        return aux_key_beat
    
    def print_debug(self):
        print()
        print("###### AUX DEBUG ######")
        print(f"Count: {len(self.aux_debug)}")
        for a in self.aux_debug:
            print(f"Index: {a[0]}")
            print(f"Ori: {a[1]} Con:{a[2]}")
            
        print("###### KEY DEBUG ######")
        print(f"Count: {len(self.key_debug)}")
        for k in self.key_debug:
            print(f"Index: {k[0]}")
            print(f"Ori: {k[1]} Con:{k[2]}")
             
class TokenConstraintor():
    def __init__(
            self,
            strct,
            in_word_pos,
            sent_form,
            tgt_dict,
            eos
    ):
        self.in_word_pos = in_word_pos
        self.tgt_dict = tgt_dict
        self.eos = eos
        self.pitch_ruler = self.PitchRuler(tgt_dict, sent_form)
        self.strct_buffer = self.StructBuffer(strct)
        self.pitch_debug = []
        self.pos_debug = []

    def pitch_constraint(self, lprobs, tokens, step):
        note_probs = torch.zeros(lprobs.size()).to(lprobs).fill_(_config.PitchPara.P_WORST.value)
        topk_idx = torch.topk(lprobs, k=_config.PitchPara.TopK.value, dim=1)
        topk_idx = topk_idx[1]

        # position of notes, to determine its upbeat or downbeat
        n_pos = self.pitch_ruler._get_num(self.tgt_dict.string(tokens[0, step : step + 1]))
        prev_note = self.pitch_ruler._get_num(self.tgt_dict.string(tokens[0, step - 3 : step - 2]))
        note_weight = _config.PitchPara.W_Middle.value

        # Update the Structure Buffer
        self.strct_buffer.update_crnt_label(step//4)
        buffer_pitch = self.strct_buffer.get_last_pitch()

        # print("IN in ")        
        
        for bidx in range(topk_idx.size(0)):
            for i in topk_idx[bidx]:
                # print("hi")
                if i == self.eos:
                    note_probs[bidx][i] = _config.PitchPara.P_BEST.value
                    continue

                curr_token = self.tgt_dict[i]
                
                # ic(curr_token)
                
                if "Pitch" not in curr_token:
                    continue

                curr_note = self.pitch_ruler._get_num(curr_token)
                
                # Pitch Constraint Part
                # violation of same pitch constraint
                dup_vio = self.pitch_ruler._check_duplicate_notes(tokens, curr_note, step)
                # violation of pitch not in key constraint
                key_vio = curr_note in _config.NOT_IN_KEY
                if dup_vio or key_vio:
                    continue

                probs = self.pitch_ruler._accept_notes(curr_note - prev_note)
                
                # Structure Constraint Part
                if curr_token in buffer_pitch:
                    probs += _config.PitchPara.P_BEST.value
                
                note_probs[bidx][i] = probs
                
        # ic(note_probs)
        top_idx = torch.topk(lprobs, k=5, dim=1)
        lprobs += note_weight * note_probs
        top1_idx = torch.topk(lprobs, k=5, dim=1)

        b_p = self.tgt_dict.string(top_idx[1]).split()
        a_p = self.tgt_dict.string(top1_idx[1]).split()

        # ic(b_p, a_p)
        # ic(top_idx[0][0].numpy(), top1_idx[0][0].numpy())
        # Update the pitch to buffer
        self.strct_buffer.update_pitch(self._get_num(a_p[0]))

        # debug info
        if b_p[0] != a_p[0]:
            pi_debug = _config.PitchDebug(
                step=step,
                note_weight=note_weight,
                prev_note=prev_note,
                beat=n_pos//4,
                before_pr=top_idx[0][0].numpy(),
                before_pi=[self.pitch_ruler._get_num(p) for p in b_p],
                after_pr=top1_idx[0][0].numpy(),
                after_pi=[self.pitch_ruler._get_num(p) for p in a_p]
            )
            self.pitch_debug.append(pi_debug)

        return lprobs

    def _get_num(self, n) -> int:
        s = n.split('_')
        return int(s[1])

    def pos_constraint(self, lprobs, tokens, step):
        def accept_pos_words(l_pos, curr_pos, prev_end):
            """
            Reduce rest notes within a word
            input:
                lpos: False, this letter is the first letter of the word and vice versa
                curr_pos: position of this token
                prev_end: position where the last note ends
            """
            if l_pos and curr_pos - prev_end == 0:
                return _config.PosPara.P_BEST.value

            elif curr_pos < prev_end:
                return _config.PosPara.P_WORST.value

            else:
                return _config.PosPara.P_MIDD.value

        """POS 主要处理句子的分段;
        lpos: 句首/其他
        """
        try:
            l_pos = self.in_word_pos[step//4]
        except IndexError:
            return lprobs

        # POS constraint
        pos_weight = _config.PosPara.W.value
        pos_align_prob = _config.PosPara.P_BEST.value

        pos_probs = torch.zeros(lprobs.size()).to(lprobs).fill_(_config.PosPara.P_WORST.value)
        topk_idx = torch.topk(lprobs, k=_config.PosPara.TopK.value, dim=1)
        topk_idx = topk_idx[1]

        prev_bar = self._get_num(self.tgt_dict.string(tokens[0, step - 4 : step - 3]))
        prev_pos = self._get_num(self.tgt_dict.string(tokens[0, step - 3 : step - 2]))
        prev_dur = self._get_num(self.tgt_dict.string(tokens[0, step - 1 : step]))
        curr_bar = self._get_num(self.tgt_dict.string(tokens[0, step : step + 1]))

        prev_pos = (prev_bar * 16) + prev_pos
        prev_end = prev_pos + prev_dur

        for bidx in range(topk_idx.size(0)):
            for i in topk_idx[bidx]:
                if i == self.eos:
                    pos_probs[bidx][i] = pos_align_prob
                    continue

                curr_token = self.tgt_dict[i]
                if "Pos" not in curr_token:
                    continue

                curr_pos = (curr_bar * 16) + self._get_num(curr_token)

                pos_probs[bidx][i] = accept_pos_words(l_pos, curr_pos, prev_end)

        top_idx = torch.topk(lprobs, k=5, dim=1)
        lprobs += pos_weight * pos_probs
        top1_idx = torch.topk(lprobs, k=5, dim=1)

        # debug info
        if top_idx[1][0][0] != top1_idx[1][0][0]:
            po_debug = _config.PositionDebug(
                step=step,
                prev_pos=prev_pos,
                prev_end=prev_end,
                curr_bar=curr_bar,
                sents_pos=l_pos,
                before_pr=top_idx[0][0].numpy(),
                before_po=self.tgt_dict.string(top_idx[1]),
                after_pr=top1_idx[0][0].numpy(),
                after_po=self.tgt_dict.string(top1_idx[1])
            )
            self.pos_debug.append(po_debug)

        return lprobs

    class PitchRuler():
        def __init__(self, tgt_dict, sent_form) -> None:
            self.tgt_dict = tgt_dict
            self.sent_form = sent_form

        def _get_num(self, n) -> int:
            s = n.split('_')
            return int(s[1])

        def _check_duplicate_notes(self, tokens, curr_note, step) -> bool:
            """Prevent too many notes with same pitch occur in a row
            """
            same_note_cnt = 0
            idx = step - 3
            while idx > 0:
                prev_note = self._get_num(self.tgt_dict.string(tokens[0, idx : idx + 1]))
                if prev_note != curr_note:
                    break
                same_note_cnt += 1
                idx -= 4

            if same_note_cnt > _config.PitchPara.Max_Same_Pitch.value:
                return True
            return False

        def _accept_notes(self, delta_notes) -> float:
            """处理倒字-根据音程范围决定分数
            """
            prob = _config.PitchPara.P_WORST.value

            # Sent form
            if (self.sent_form == _config.Form.ascend and delta_notes > 0) or \
                (self.sent_form == _config.Form.descend and delta_notes < 0):
                print("Sent Form Right")
                prob = _config.PitchPara.Sent_form_bias.value

            return prob

    class StructBuffer():
        crnt_step = None
        crnt_unit = None
        crnt_idx = 0

        def __init__(self, strct_label):
            self.structs = sum([ [l[0]] * l[1]  for l in strct_label ], [])
            self.strct_labels = set(strct_label)
            self.buffer = { l[0]: self.StructUnit(l) for l in self.strct_labels }

        def update_crnt_label(self, step):
            label = self.structs[step]
            # print('hi step', step, label)
            if self.crnt_step == None or self.structs[self.crnt_step] != label:
                self.crnt_unit = self.buffer[label]
                self.crnt_idx = 0

            if self.crnt_step != step:
                self.crnt_idx += 1

        def get_last_pitch(self):
            buffer_pitch = []
            if self.crnt_unit.finished:
                last_pitch = self.crnt_unit.pitch[self.crnt_idx]
                buffer_pitch.append(last_pitch)

                if last_pitch >= 64: # higher than E4
                    buffer_pitch.append(last_pitch-12)

                if last_pitch <= 69: # lower than A4
                    buffer_pitch.append(last_pitch+12)

            return buffer_pitch

        def get_last_pos(self):
            return self.crnt_unit.position[self.crnt_idx] if self.crnt_unit.finished else None

        def get_last_dur(self):
            return self.crnt_unit.duration[self.crnt_idx] if self.crnt_unit.finished else None

        def update_finished(self):
            max_length = self.crnt_unit.word_num
            if len(self.crnt_unit.pitch) == max_length:
            # and \
            # len(self.crnt_unit.position) == max_length and \
            # len(self.crnt_unit.duration) == max_length:
                self.crnt_unit.finished = True

        def update_pitch(self, pitch):
            if not self.crnt_unit.finished:
                # print('not finished', pitch)
                self.crnt_unit.pitch.append(pitch)
                self.update_finished()

        def update_pos(self, pos):
            if not self.crnt_unit.finished:
                self.crnt_unit.pitch.append(pos)
                self.update_finished()

        def update_dur(self, dur):
            if not self.crnt_unit.finished:
                self.crnt_unit.pitch.append(dur)
                self.update_finished()

        class StructUnit():
            def __init__(self, strct_label):
                self.label = strct_label[0]
                self.word_num = strct_label[1]
                self.finished = False
                self.pitch = []
                self.position = []
                self.duration = []
    
    def print_debug(self):
        # print out debug info
        print()
        print("#### Total Modified counts ####")
        print("Pitch: ", len(self.pitch_debug))
        print("Pos: ", len(self.pos_debug))
        
        if _config.PIT_DEBUG and len(self.pitch_debug) > 0:
            print("-- Pitch Constraint Debug Info --")
            for pi in self.pitch_debug:
                print(pi)
                print()
        if _config.POS_DEBUG and len(self.pos_debug) > 0:
            print("-- Position Constraint Debug Info --")
            for p in self.pos_debug:
                print(p)
                print()
        