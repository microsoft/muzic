from .raw_unit_base import RawUnitBase


class RawUnitI1(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path, pos_info, bars_positions, bars_chords, bars_insts,
        bar_begin, bar_end, **kwargs
    ):
        """
        抽取使用的乐器。
        :return:
        - tuple，使用到的乐器的ID。无None。
        """
        insts = set()
        for bar_insts in bars_insts[bar_begin: bar_end]:
            for inst_id in bar_insts:
                insts.add(inst_id)

        insts = tuple(insts)
        return insts


class RawUnitI2(RawUnitBase):
    """
    - tuple, 前半段使用的乐器，当bar数量为非正偶数的时候返回None
    - tuple，后半段使用的乐器，当bar数量为非正偶数的时候返回None
    """
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path, pos_info, bars_positions, bars_chords, bars_insts,
        bar_begin, bar_end, **kwargs
    ):
        num_bars = bar_end - bar_begin
        if num_bars <= 0 or num_bars % 2 == 1:
            return None, None

        left_insts = set()
        right_insts = set()
        for bar_insts in bars_insts[bar_begin: bar_begin + num_bars // 2]:
            for inst_id in bar_insts:
                left_insts.add(inst_id)
        for bar_insts in bars_insts[bar_begin + num_bars // 2: bar_end]:
            for inst_id in bar_insts:
                right_insts.add(inst_id)

        left_insts = tuple(left_insts)
        right_insts = tuple(right_insts)

        return left_insts, right_insts
