from .raw_unit_base import RawUnitBase


class RawUnitTS1(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
        - list, 所用过的所有ts，每个元素是一个元组，例如(3, 4)
        """
        ts_set = set()
        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]
        assert pos_info[begin][1] is not None
        for idx in range(begin, end):
            ts = pos_info[idx][1]
            if ts is None:
                continue
            ts_set.add(ts)
        return list(ts_set)
