from .raw_unit_base import RawUnitBase

from ..utils.data import convert_dict_key_to_str, convert_dict_key_to_int


class RawUnitP1(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - int，最低pitch，不考虑鼓。没有音符则返回None
        """
        low = 1000

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        no_notes = True
        for idx in range(begin, end):
            pos_item = pos_info[idx]
            insts_notes = pos_item[4]
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]
                for pitch, _, _ in inst_notes:
                    low = min(low, pitch)
                    no_notes = False

        if no_notes:
            return None
        return low


class RawUnitP2(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - int，最高pitch，不考虑鼓。没有音符则返回None
        """
        high = -1

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        no_notes = True
        for idx in range(begin, end):
            pos_item = pos_info[idx]
            insts_notes = pos_item[4]
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]
                for pitch, _, _ in inst_notes:
                    high = max(high, pitch)
                    no_notes = False
        if no_notes:
            return None
        return high


class RawUnitP3(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path, pos_info, bars_positions, bars_chords, bars_insts,
        bar_begin, bar_end, **kwargs
    ):
        """
        各乐器的总音高（pitch之和），不计算鼓
        :return:
        - dict: 各乐器的pitch之和, key为inst id，value为pitch的和。如果无除鼓外的其他乐器的音符，则返回空dict
        """
        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        pitch_record = {}
        for idx in range(begin, end):
            pos_item = pos_info[idx]
            insts_notes = pos_item[4]
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]
                if inst_id not in pitch_record:
                    pitch_record[inst_id] = 0
                for pitch, _, _ in inst_notes:
                    pitch_record[inst_id] += pitch

        return pitch_record

    @classmethod
    def repr_value(cls, value):
        return convert_dict_key_to_str(value)

    @classmethod
    def derepr_value(cls, rep_value):
        return convert_dict_key_to_int(rep_value)
