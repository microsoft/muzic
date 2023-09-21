from .raw_unit_base import RawUnitBase

from ..utils.data import convert_dict_key_to_str, convert_dict_key_with_eval


class RawUnitR1(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - int, 片段的onset总数，无note则为0
        """

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        num_onsets = 0
        for pos_item in pos_info[begin: end]:
            insts_notes = pos_item[-1]
            if insts_notes is not None:
                num_onsets += 1

        return num_onsets


class RawUnitR2(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - float: 片段的beat总数
        """

        pos_resolution = 12
        assert pos_resolution == encoder.vm.pos_resolution, str(encoder.vm.pos_resolution)

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]
        num_beats = (end - begin) / pos_resolution

        return num_beats


class RawUnitR3(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - float: num_onsets / num_beats，即note density
        """

        num_onsets = RawUnitR1.extract(
            encoder, midi_dir, midi_path,
            pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
        )
        num_beats = RawUnitR2.extract(
            encoder, midi_dir, midi_path,
            pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
        )
        density = num_onsets / num_beats

        return density


class RawUnitR4(RawUnitBase):
    """
    鼓在各local pos上的音符数量
    """
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - dict: key为TS，value为list，list的长度为该TS情况下，每个小节的pos数量，list的每个元素为鼓在该pos上的音符数量。
                    如果没有鼓，value为None
        """
        r = {}
        r_has_drum = {}

        for bar_idx in range(bar_begin, bar_end):
            begin, end = bars_positions[bar_idx]
            num_bar_pos = end - begin
            ts = pos_info[begin][1]
            assert ts is not None
            if ts not in r:
                r[ts] = [0] * num_bar_pos
                r_has_drum[ts] = False
            for pos_item in pos_info[begin: end]:
                insts_notes = pos_item[-1]
                if insts_notes is None:
                    continue
                local_pos = pos_item[2]
                for inst_id in insts_notes:
                    if inst_id != 128:
                        continue
                    inst_notes = insts_notes[inst_id]
                    num_notes = len(inst_notes)
                    r[ts][local_pos] += num_notes
                    r_has_drum[ts] = True

        for ts in r_has_drum:
            if not r_has_drum[ts]:
                r[ts] = None

        return r

    @classmethod
    def repr_value(cls, value):
        return convert_dict_key_to_str(value)

    @classmethod
    def derepr_value(cls, rep_value):
        return convert_dict_key_with_eval(rep_value)
