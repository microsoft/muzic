# Author: Botao Yu

import os
import math

from . import const
from . import vocab_config


class VocabManager:
    def __init__(
        self,
        pos_resolution=vocab_config.pos_resolution,
        max_ts_denominator_power=vocab_config.max_ts_denominator_power,
        max_notes_per_bar=vocab_config.max_notes_per_bar,
        tempo_quant=vocab_config.tempo_quant,
        min_tempo=vocab_config.min_tempo,
        max_tempo=vocab_config.max_tempo,
        velocity_quant=vocab_config.velocity_quant,
        max_duration=vocab_config.max_duration,
        max_bar_num=vocab_config.max_bar_num,
    ):

        self.pos_resolution = pos_resolution  # per beat (quarter note)

        self.max_ts_denominator_power = max_ts_denominator_power  # x/1 x/2 x/4 ... x/64
        self.max_ts_denominator = 2 ** self.max_ts_denominator_power
        self.max_notes_per_bar = max_notes_per_bar  # max number of whole notes within a bar

        self.tempo_quant = tempo_quant  # 2 ** (1 / 12)
        self.min_tempo = min_tempo
        self.max_tempo = max_tempo

        self.velocity_quant = velocity_quant

        self.max_duration = max_duration  # 2 ** 8 * beat

        self.beat_note_factor = 4  # In midi format a note is always 4 beats

        self.max_bar_num = max_bar_num

        # ===== Generating Vocabs =====
        self.ts_dict, self.ts_list = self.generate_ts_vocab(self.max_ts_denominator_power, self.max_notes_per_bar)
        self.dur_enc, self.dur_dec = self.generate_duration_vocab(self.max_duration, self.pos_resolution)

        self.vocab = self.generate_vocab()
        
    def vocab_to_str_list(self):
        return ['%s-%d' % (item[0], item[1]) for item in self.vocab]

    def dump_vocab(self, file_path, fairseq_dict=False):
        vocab_str_list = self.vocab_to_str_list()
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for word in vocab_str_list:
                if fairseq_dict:
                    line = '%s 1\n' % word
                else:
                    line = '%s\n' % word
                f.write(line)

    def generate_vocab(self):
        vocab = []

        for bar_idx in range(self.max_bar_num):
            vocab.append((const.BAR_ABBR, bar_idx))

        for idx in range(self.beat_note_factor * self.max_notes_per_bar * self.pos_resolution):
            vocab.append((const.POS_ABBR, idx))

        for idx in range(129):
            vocab.append((const.INST_ABBR, idx))

        for idx in range(256):
            vocab.append((const.PITCH_ABBR, idx))

        for idx in range(len(self.dur_dec)):
            vocab.append((const.DURATION_ABBR, idx))

        for idx in range(self.convert_vel_to_id(127) + 1):
            vocab.append((const.VELOCITY_ABBR, idx))

        for idx in range(len(self.ts_list)):
            vocab.append((const.TS_ABBR, idx))

        for idx in range(self.convert_tempo_to_id(self.max_tempo) + 1):
            vocab.append((const.TEMPO_ABBR, idx))

        return vocab

    def reduce_time_signature(self, numerator, denominator):
        while denominator > self.max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
            denominator //= 2
            numerator //= 2
        # decomposition (when length of a bar exceed max_notes_per_bar)
        while numerator > self.max_notes_per_bar * denominator:
            for i in range(2, numerator + 1):
                if numerator % i == 0:
                    numerator //= i
                    break
        return numerator, denominator

    @staticmethod
    def generate_ts_vocab(max_ts_denominator_power, max_notes_per_bar):
        ts_dict = dict()
        ts_list = list()
        for i in range(0, max_ts_denominator_power + 1):  # 1 ~ 64
            for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
                ts_dict[(j, 2 ** i)] = len(ts_dict)
                ts_list.append((j, 2 ** i))
        return ts_dict, ts_list

    @staticmethod
    def generate_duration_vocab(max_duration, pos_resolution):
        dur_enc = list()
        dur_dec = list()
        for i in range(max_duration):
            for j in range(pos_resolution):
                dur_dec.append(len(dur_enc))
                for k in range(2 ** i):
                    dur_enc.append(len(dur_dec) - 1)
        return dur_enc, dur_dec

    def convert_ts_to_id(self, x):
        return self.ts_dict[x]

    def convert_id_to_ts(self, x):
        return self.ts_list[x]

    def convert_tempo_to_id(self, x):
        x = max(x, self.min_tempo)
        x = min(x, self.max_tempo)
        x = x / self.min_tempo
        e = round(math.log2(x) * self.tempo_quant)
        return e

    def convert_id_to_tempo(self, x):
        return 2 ** (x / self.tempo_quant) * self.min_tempo

    def convert_pitch_to_id(self, x, is_drum=False):
        if is_drum:
            return int(x + 128)
        return int(x)

    def convert_id_to_pitch(self, x):
        if x >= 128:
            x = x - 128
        return x

    def convert_vel_to_id(self, x):
        return int(x // self.velocity_quant)

    def convert_id_to_vel(self, x):
        return (x * self.velocity_quant) + (self.velocity_quant // 2)

    def convert_dur_to_id(self, x):
        return int(self.dur_enc[x] if x < len(self.dur_enc) else self.dur_enc[-1])

    def convert_id_to_dur(self, x, min_pos=1):
        r = self.dur_dec[x] if x < len(self.dur_dec) else self.dur_dec[-1]
        if min_pos is not None:
            r = max(r, min_pos)
        return r

    def time_to_pos(self, t, ticks_per_beat):
        return round(t * self.pos_resolution / ticks_per_beat)

    def pos_to_time(self, pos, ticks_per_beat, pos_resolution=None):
        if pos_resolution is None:
            pos_resolution = self.pos_resolution
        return round(pos * ticks_per_beat / pos_resolution)
