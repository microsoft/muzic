from .raw_unit_base import RawUnitBase

from ..utils.data import convert_dict_key_to_str, convert_dict_key_to_int


class RawUnitN2(RawUnitBase):
    """

    """
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path, pos_info, bars_positions, bars_chords, bars_insts,
        bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - dict, 各乐器的音符数量
        """
        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        num_note_record = {}
        for idx in range(begin, end):
            pos_item = pos_info[idx]
            insts_notes = pos_item[4]
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                inst_notes = insts_notes[inst_id]
                if inst_id not in num_note_record:
                    num_note_record[inst_id] = 0
                num_note_record[inst_id] += len(inst_notes)

        return num_note_record

    @classmethod
    def repr_value(cls, value):
        return convert_dict_key_to_str(value)

    @classmethod
    def derepr_value(cls, rep_value):
        return convert_dict_key_to_int(rep_value)

