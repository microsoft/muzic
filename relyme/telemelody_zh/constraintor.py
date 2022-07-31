import re
import torch
from pypinyin import lazy_pinyin, Style

import config as _config

PUNC = [",", "."]
TOKEN = ['Bar', 'Pos', 'Pitch', 'Dur']

class TokenConstraintor():
    def __init__(
            self,
            lyrics,
            strct,
            letter_pos,
            sent_form,
            note_range,
            tgt_dict,
            eos
    ):
        self.lyrics = lyrics
        self.letter_pos = letter_pos
        self.tgt_dict = tgt_dict
        self.eos = eos

        self.pitch_ruler = self.PitchRuler(tgt_dict, lyrics, sent_form, note_range)
        self.strct_buffer = self.StructBuffer(strct)

    def pitch_constraint(self, lprobs, tokens, step, c_cnt):
        def find_han(s):
            regex = r"[\u4e00-\u9fff]"
            matches = re.findall(regex, s, re.UNICODE)
            return matches

        try:
            curr_word, sents_pos = self.lyrics[ step // 4 ]

        except IndexError:
            return lprobs

        note_probs = torch.zeros(lprobs.size()).to(lprobs).fill_(_config.PitchPara.P_WORST.value)
        topk_idx = torch.topk(lprobs, k=_config.PitchPara.TopK.value, dim=1)
        topk_idx = topk_idx[1]

        prev_word = self.lyrics[(step//4) - 1][0]
        py = lazy_pinyin(prev_word + curr_word, style=Style.TONE3)
        front = self.pitch_ruler._get_tone_id(py[-2][-1])
        back = self.pitch_ruler._get_tone_id(py[-1][-1])

        # position of notes, to determine its upbeat or downbeat
        n_pos = self.pitch_ruler._get_num(self.tgt_dict.string(tokens[0, step : step + 1]))
        prev_note = self.pitch_ruler._get_num(self.tgt_dict.string(tokens[0, step - 3 : step - 2]))
        note_weight = self.pitch_ruler._get_note_weights(sents_pos, n_pos)

        if len(find_han(curr_word)) * len(find_han(prev_word)) == 0:
            # or sents_pos == _config.Sent.HEAD or:
            # in the first word of the sent or not chinese
            return lprobs

        # Update the Structure Buffer
        self.strct_buffer.update_crnt_label(step//4)
        buffer_pitch = self.strct_buffer.get_last_pitch()

        for bidx in range(topk_idx.size(0)):
            for i in topk_idx[bidx]:
                if i == self.eos:
                    note_probs[bidx][i] = _config.PitchPara.P_BEST.value
                    continue

                curr_token = self.tgt_dict[i]
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

                # probs = _config.PitchPara.P_WORST.value
                probs = self.pitch_ruler._accept_notes(curr_note - prev_note, front, back)
                
                Structure Constraint Part
                if curr_token in buffer_pitch:
                    probs += _config.PitchPara.P_BEST.value
                
                note_probs[bidx][i] = probs
        
        top_idx = torch.topk(lprobs, k=5, dim=1)
        lprobs += note_weight * note_probs
        top1_idx = torch.topk(lprobs, k=5, dim=1)

        b_p = self.tgt_dict.string(top_idx[1]).split()
        a_p = self.tgt_dict.string(top1_idx[1]).split()

        
        # Update the pitch to buffer
        self.strct_buffer.update_pitch(self._get_num(a_p[0]))

        # debug info
        if b_p[0] != a_p[0]:
            pi_debug = _config.PitchDebug(
                step=step,
                note_weight=note_weight,
                curr_word=curr_word,
                prev_word=prev_word,
                prev_note=prev_note,
                beat=n_pos//4,
                sents_flag=sents_pos,
                before_pr=top_idx[0][0].numpy(),
                before_pi=[self.pitch_ruler._get_num(p) for p in b_p],
                after_pr=top1_idx[0][0].numpy(),
                after_pi=[self.pitch_ruler._get_num(p) for p in a_p]
            )
            c_cnt.append(pi_debug)

        return lprobs

    def _get_num(self, n) -> int:
        s = n.split('_')
        return int(s[1])

    def pos_constraint(self, lprobs, tokens, step, p_cnt):
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
            l_pos = self.letter_pos[step//4]
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
                curr_word=self.lyrics[step//4][0],
                prev_pos=prev_pos,
                prev_end=prev_end,
                curr_bar=curr_bar,
                sents_pos=l_pos,
                before_pr=top_idx[0][0].numpy(),
                before_po=self.tgt_dict.string(top_idx[1]),
                after_pr=top1_idx[0][0].numpy(),
                after_po=self.tgt_dict.string(top1_idx[1])
            )
            p_cnt.append(po_debug)

        return lprobs

    class PitchRuler():
        def __init__(self, tgt_dict, lyrics, sent_form, note_range) -> None:
            self.tgt_dict = tgt_dict
            self.lyrics = lyrics
            self.sent_form = sent_form
            self.note_range = note_range
            self.interval_range = _config.interval_range

        def _get_num(self, n) -> int:
            s = n.split('_')
            return int(s[1])

        def _get_tone_id(self, s) -> int:
            if s in ["1", "2", "3", "4"]:
                return int(s) - 1
            return 4

        def _get_note_weights(self, sents_pos, n_pos: int = 0) -> float:
            """ _get_note_weight return the notes_weights according to 严格 比较严格 比较自由

            Args:
                sents_pos: whether the lyric is in the first or last sentence in the structure
                pos: the next lyric position, if its punctuation, then current word is th last word
            """
            strict_cond = ( sents_pos == _config.Sent.LAST or
                            sents_pos == _config.Strct.FIRST or
                            sents_pos == _config.Strct.LAST )
            # strict
            if strict_cond:
                return _config.PitchPara.W_Strict.value

            # not so strict
            elif (n_pos / 4) % 2 == 0:
                return _config.PitchPara.W_Middle.value

            # free
            else:
                return _config.PitchPara.W_Looose.value

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

        def _accept_notes(self, delta_notes, front, back) -> float:
            """处理倒字-根据音程范围决定分数
            """
            note_r = self.interval_range[front][back]
            note_r = [ [ n * self.note_range for n in r ] for r in note_r ]

            prob = _config.PitchPara.P_WORST.value

            # Between Notes
            if delta_notes >= note_r[0][0] and delta_notes <= note_r[0][1]:
                prob = _config.PitchPara.P_BEST.value

            elif delta_notes >= note_r[1][0] and delta_notes <= note_r[1][1]:
                prob = _config.PitchPara.P_SECO.value

            elif delta_notes >= note_r[2][0] and delta_notes <= note_r[2][1]:
                prob = _config.PitchPara.P_THIR.value

            # Sent form
            if (self.sent_form == _config.Form.ascend and delta_notes > 0) or \
                (self.sent_form == _config.Form.descend and delta_notes < 0):
                prob += _config.PitchPara.Sent_form_bias.value

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