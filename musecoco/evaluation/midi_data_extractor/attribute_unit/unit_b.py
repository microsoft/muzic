from .unit_base import UnitBase

from .raw_unit_b import RawUnitB1


class UnitB1(UnitBase):
    """
    抽取bar的个数
    """

    @property
    def version(self) -> str:
        return 'v1.0'

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

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * 14
        if not use:
            vector[-1] = 1
            return vector
        vector[value - 4] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 14


class UnitB1s1(UnitBase):
    """
    抽取bar的个数
    """

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitB1

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """
        :return:
            - int: bar的个数
            - int: bar个数区间的id：0：1-4，1：5-8，2：9-12，3：13-16
            无None
        """
        num_bars = raw_data['B1']
        if not (0 < num_bars <= 16):
            # raise NotImplementedError("The current implementation only supports 1~16 bars.")
            return num_bars, -1
        bar_id = cls.convert_num_bars_to_id(num_bars)
        return num_bars, bar_id

    @classmethod
    def convert_num_bars_to_id(cls, num_bars):
        return int(max(num_bars - 1, 0) / 4)

    def get_vector(self, use=True, use_info=None):
        # 顺序：0， 1， 2， 3， NA
        _, bar_id = self.value
        vector = [0] * self.vector_dim
        if not use:
            vector[-1] = 1
            return vector
        vector[bar_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 5
