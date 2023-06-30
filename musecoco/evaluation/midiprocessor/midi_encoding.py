import os
import typing
import pickle
from . import midi_utils
from copy import deepcopy

from . import const
from .vocab_manager import VocabManager
from . import data_utils
from . import keys_normalization


ENCODINGS = ('REMIGEN', 'REMIGEN2')


def raise_encoding_method_error(encoding_method):
    raise ValueError("Encoding method %s is not supported." % encoding_method)


def check_encoding_method(encoding_method):
    assert encoding_method in ENCODINGS, "Encoding method %s not in the supported: %s" % \
                                               (encoding_method, ', '.join(ENCODINGS))


class MidiEncoder(object):
    def __init__(self, encoding_method):
        # ===== Check =====
        check_encoding_method(encoding_method)

        # ===== Authorized =====
        self.encoding_method = encoding_method
        self.vm = VocabManager()

        with open(os.path.join(os.path.dirname(__file__), const.KEY_PROFILE), 'rb') as f:
            self.key_profile = pickle.load(f)

    # ===== Basic Functions ===
    def time_to_pos(self, *args, **kwargs):
        return self.vm.time_to_pos(*args, **kwargs)

    def collect_pos_info(self, midi_obj, trunc_pos=None, tracks=None, remove_same_notes=False, end_offset=0):
        if tracks is not None:
            from collections.abc import Iterable
            assert isinstance(tracks, (int, Iterable))
            if isinstance(tracks, str):
                tracks = int(tracks)
            if isinstance(tracks, int):
                if tracks < 0:
                    tracks = len(midi_obj.instruments) + tracks
                tracks = (tracks,)

        max_pos = 0
        for inst in midi_obj.instruments:
            for note in inst.notes:
                pos = self.time_to_pos(note.start, midi_obj.ticks_per_beat)
                max_pos = max(max_pos, pos)
        max_pos = max_pos + 1  # 最大global position
        if trunc_pos is not None:
            max_pos = min(max_pos, trunc_pos)

        pos_info = [
            [None, None, None, None, None]  # (bar, ts, local_pos, tempo, insts_notes)
            for _ in range(max_pos)
        ]
        pos_info: typing.List
        # bar: every pos
        # ts: only at pos where it changes, otherwise None
        # local_pos: every pos
        # tempo: only at pos where it changes, otherwise None
        # insts_notes: only at pos where the note starts, otherwise None

        ts_changes = midi_obj.time_signature_changes
        zero_pos_ts_change = False
        for ts_change in ts_changes:
            pos = self.time_to_pos(ts_change.time, midi_obj.ticks_per_beat)
            if pos >= max_pos:
                continue
            if pos == 0:
                zero_pos_ts_change = True
            ts_numerator = int(ts_change.numerator)
            ts_denominator = int(ts_change.denominator)
            # if self.ignore_ts:
            #     assert (ts_numerator, ts_denominator) == const.DEFAULT_TS
            ts_numerator, ts_denominator = self.vm.reduce_time_signature(ts_numerator, ts_denominator)
            pos_info[pos][1] = (ts_numerator, ts_denominator)
        if not zero_pos_ts_change:
            pos_info[0][1] = const.DEFAULT_TS

        tempo_changes = midi_obj.tempo_changes
        zero_pos_tempo_change = False
        for tempo_change in tempo_changes:
            pos = self.time_to_pos(tempo_change.time, midi_obj.ticks_per_beat)
            if pos >= max_pos:
                continue
            if pos == 0:
                zero_pos_tempo_change = True
            pos_info[pos][3] = tempo_change.tempo
        if not zero_pos_tempo_change:
            pos_info[0][3] = const.DEFAULT_TEMPO

        insts = midi_obj.instruments
        for inst_idx, inst in enumerate(insts):
            if tracks is not None and inst_idx not in tracks:
                continue
            # if self.ignore_insts:
            #     inst_id = 0
            # else:
            inst_id = 128 if inst.is_drum else int(inst.program)
            notes = inst.notes
            for note in notes:
                pitch = int(note.pitch)
                velocity = int(note.velocity)
                start_time = int(note.start)
                end_time = int(note.end + end_offset)
                assert end_time > start_time
                pos_start = self.time_to_pos(start_time, midi_obj.ticks_per_beat)
                pos_end = self.time_to_pos(end_time, midi_obj.ticks_per_beat)
                duration = pos_end - pos_start

                if pos_info[pos_start][4] is None:
                    pos_info[pos_start][4] = dict()
                if inst_id not in pos_info[pos_start][4]:
                    pos_info[pos_start][4][inst_id] = []
                note_info = [pitch, duration, velocity]
                if remove_same_notes:
                    if note_info in pos_info[pos_start][4][inst_id]:
                        continue
                pos_info[pos_start][4][inst_id].append([pitch, duration, velocity])

        cnt = 0
        bar = 0
        measure_length = None
        ts = const.DEFAULT_TS  # default MIDI time signature
        for j in range(max_pos):
            now_ts = pos_info[j][1]
            if now_ts is not None:
                if now_ts != ts:
                    ts = now_ts
            if cnt == 0:
                measure_length = ts[0] * self.vm.beat_note_factor * self.vm.pos_resolution // ts[1]
            pos_info[j][0] = bar
            pos_info[j][2] = cnt
            cnt += 1
            if cnt >= measure_length:
                assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(j)
                cnt = 0
                bar += 1

        return pos_info

    def remove_empty_bars_for_pos_info(self, pos_info):
        cur_ts = None
        cur_tempo = None
        first_valid_bar_idx = None
        for cur_pos in range(len(pos_info)):
            pos_item = pos_info[cur_pos]
            if pos_item[-1] is not None:
                first_valid_bar_idx = pos_item[0]
                break
        offset = first_valid_bar_idx
        first_valid_pos = None
        for cur_pos in range(len(pos_info)):
            pos_item = pos_info[cur_pos]
            if pos_item[1] is not None:
                cur_ts = pos_item[1]
            if pos_item[3] is not None:
                cur_tempo = pos_item[3]
            if pos_item[0] == first_valid_bar_idx:
                first_valid_pos = cur_pos
                break

        last_valid_pos = None
        for cur_pos in range(len(pos_info) - 1, -1, -1):
            pos_item = pos_info[cur_pos]
            if pos_item[-1] is not None:
                last_valid_pos = cur_pos
                break
        pos_info = pos_info[first_valid_pos: last_valid_pos + 1]
        pos_info[0][1] = cur_ts if cur_ts is not None else (4, 4)
        pos_info[0][3] = cur_tempo if cur_tempo is not None else 120.0

        for pos_item in pos_info:
            pos_item[0] -= offset

        return pos_info

    def convert_pos_info_to_pos_info_id(self, pos_info):
        pos_info_id = deepcopy(pos_info)
        # (bar, ts, local_pos, tempo, insts_notes)

        for idx, item in enumerate(pos_info_id):
            bar, ts, local_pos, tempo, insts_notes = item
            if ts is not None:
                ts_id = self.vm.convert_ts_to_id(ts)
                item[1] = ts_id
            if tempo is not None:
                tempo_id = self.vm.convert_tempo_to_id(tempo)
                item[3] = tempo_id
            if insts_notes is not None:
                for inst_id in insts_notes:
                    inst_notes = insts_notes[inst_id]
                    for inst_note in inst_notes:
                        # (pitch, duration, velocity)
                        pitch, duration, velocity = inst_note
                        pitch_id = self.vm.convert_pitch_to_id(pitch, is_drum=inst_id == 128)
                        duration_id = self.vm.convert_dur_to_id(duration)
                        velocity_id = self.vm.convert_vel_to_id(velocity)
                        inst_note[0] = pitch_id
                        inst_note[1] = duration_id
                        inst_note[2] = velocity_id
        return pos_info_id

    def convert_pos_info_id_to_pos_info(self, pos_info_id):
        pos_info = deepcopy(pos_info_id)
        # (bar, ts_id, local_pos, tempo_id, insts_notes)

        for idx, item in enumerate(pos_info):
            bar, ts_id, local_pos, tempo_id, insts_notes = item
            if ts_id is not None:
                ts = self.vm.convert_id_to_ts(ts_id)
                item[1] = ts
            if tempo_id is not None:
                tempo = self.vm.convert_id_to_tempo(tempo_id)
                item[3] = tempo
            if insts_notes is not None:
                # (pitch, duration, velocity, pos_end)
                for inst_id in insts_notes:
                    inst_notes = insts_notes[inst_id]
                    for inst_note in inst_notes:
                        pitch, duration, velocity = inst_note
                        pitch = self.vm.convert_id_to_pitch(pitch)
                        duration = self.vm.convert_id_to_dur(duration)
                        velocity = self.vm.convert_id_to_vel(velocity)
                        inst_note[0] = pitch
                        inst_note[1] = duration
                        inst_note[2] = velocity
        return pos_info

    def encode_file(
        self,
        file_path,
        midi_checker='default',
        midi_obj=None,
        end_offset=0,
        normalize_pitch_value=False,
        trunc_pos=None,
        tracks=None,
        save_path=None,
        save_pos_info_id_path=None,
        **kwargs
    ):
        encoding_method = self.encoding_method

        if midi_obj is None:
            midi_obj = midi_utils.load_midi(file_path, midi_checker=midi_checker)

        pos_info = self.collect_pos_info(midi_obj, trunc_pos=trunc_pos, tracks=tracks, end_offset=end_offset)

        if normalize_pitch_value:
            pos_info, is_major, pitch_shift = self.normalize_pitch(pos_info)

        pos_info_id = self.convert_pos_info_to_pos_info_id(pos_info)
        if save_pos_info_id_path is not None:
            data_utils.json_save(pos_info_id, save_pos_info_id_path)

        token_lists = None
        if encoding_method == 'REMI':  # Todo: REMI encoding参考原版重写
            # from . import enc_remi_utils
            # token_lists = enc_remi_utils.convert_pos_info_to_remi_token_lists(
            #     pos_info_id,
            #     **kwargs
            # )
            raise NotImplementedError("Need to rewrite REMI encoding")
        elif encoding_method == 'REMIGEN':
            from . import enc_remigen_utils
            token_lists = enc_remigen_utils.convert_pos_info_to_token_lists(
                pos_info_id,
                **kwargs
            )
        elif encoding_method == 'REMIGEN2':
            from . import enc_remigen2_utils
            token_lists = enc_remigen2_utils.convert_pos_info_to_token_lists(
                pos_info_id,
                **kwargs
            )
        elif encoding_method == 'STACKED':
            raise NotImplementedError
            # from . import enc_stacked_utils
            # token_lists = enc_stacked_utils.convert_pos_info_id_to_token_lists(
            #     pos_info_id,
            #     **kwargs
            # )
        elif encoding_method == 'CP2':
            raise NotImplementedError
            # from . import enc_cp2_utils
            # token_lists = enc_cp2_utils.convert_pos_info_id_to_token_lists(
            #     pos_info_id,
            #     **kwargs
            # )
        else:
            raise_encoding_method_error(encoding_method)

        if save_path is not None:
            try:
                self.dump_token_lists(token_lists, save_path, no_internal_blanks=True)
            except IOError:
                print("Wrong! Saving failed: \nMIDI: %s\nSave Path: %s" % file_path, save_path)

        return token_lists

    def normalize_pitch(self, pos_info, inplace=False):
        assert self.key_profile is not None, "Please load key_profile first, using load_key_profile method."
        pitch_shift, is_major, _, _ = keys_normalization.get_pitch_shift(
            pos_info, self.key_profile,
            normalize=True, use_duration=True, use_velocity=True,
            ensure_valid_range=True
        )
        pitch_shift = int(pitch_shift)
        if not inplace:
            pos_info = deepcopy(pos_info)
        for bar, ts, pos, tempo, insts_notes in pos_info:
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]
                for note_idx, (pitch, duration, velocity) in enumerate(inst_notes):
                    # inst_notes[note_idx] = (pitch + pitch_shift, duration, velocity)
                    inst_notes[note_idx][0] = pitch + pitch_shift
        return pos_info, is_major, pitch_shift

    # Finished
    def convert_token_lists_to_token_str_lists(self, token_lists):
        """
        将一个文件的encoding token_lists（二层列表）转换为str lists
        :param token_lists:
        :return:
        """
        encoding_method = self.encoding_method
        if encoding_method == 'REMI':
            # from . import enc_remi_utils
            # return enc_remi_utils.convert_remi_token_lists_to_token_str_lists(token_lists)
            raise NotImplementedError
        elif encoding_method == 'REMIGEN':
            from . import enc_remigen_utils
            return enc_remigen_utils.convert_remigen_token_lists_to_token_str_lists(token_lists)
        elif encoding_method == 'REMIGEN2':
            from . import enc_remigen2_utils
            return enc_remigen2_utils.convert_remigen_token_lists_to_token_str_lists(token_lists)
        elif encoding_method == 'STACKED':
            # from . import enc_stacked_utils
            # return enc_stacked_utils.convert_token_lists_to_token_str_lists(token_lists)
            raise NotImplementedError
        elif encoding_method == 'CP2':
            # from . import enc_cp2_utils
            # return enc_cp2_utils.convert_token_lists_to_token_str_lists(token_lists)
            raise NotImplementedError
        else:
            raise_encoding_method_error(encoding_method)

    # Finished
    def dump_token_lists(self, token_lists, file_path, **kwargs):
        """
        将一个文件的encoding token_lists转换成str并存为文件
        :param token_lists:
        :param file_path:
        :return:
        """
        token_str_lists = self.convert_token_lists_to_token_str_lists(token_lists)
        data_utils.dump_lists(token_str_lists, file_path, **kwargs)
