from .raw_unit_base import RawUnitBase


class RawUnitB1(RawUnitBase):
    """
    抽取bar的个数
    """

    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            int，bar个数
        """
        return bar_end - bar_begin
