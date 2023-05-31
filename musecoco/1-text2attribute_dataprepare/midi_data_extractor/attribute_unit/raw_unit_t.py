from .raw_unit_base import RawUnitBase


class RawUnitT1(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - tuple of float: 所使用的所有tempo，已去重
        """
        tempo_set = set()
        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]
        assert pos_info[begin][3] is not None
        for idx in range(begin, end):
            tempo = pos_info[idx][3]
            if tempo is None:
                continue
            tempo_set.add(tempo)
        tempo_set = tuple(tempo_set)
        return tempo_set
