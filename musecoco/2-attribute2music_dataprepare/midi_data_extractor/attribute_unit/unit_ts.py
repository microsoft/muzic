from .unit_base import UnitBase
from .raw_unit_ts import RawUnitTS1


class UnitTS1(UnitBase):
    """
    所用过的唯一ts
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
            所用过的唯一ts，元组，例如(3, 4)代表3/4拍，如果不止一个ts，则return None
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
        if len(ts_set) > 1:
            return None
        return list(ts_set)[0]

    def get_vector(self, use=True, use_info=None):
        ts = self.value
        vector = [0] * (len(self.encoder.vm.ts_list) + 1)
        if not use or ts is None:
            vector[-1] = 1
            return vector
        ts_id = self.encoder.vm.convert_ts_to_id(ts)
        vector[ts_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return len(self.encoder.vm.ts_list) + 1


class UnitTS1s1(UnitBase):
    """
    所用过的唯一ts，常见类型+其他
    """
    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitTS1

    @classmethod
    def convert_raw_to_value(
        cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        ts_set = raw_data['TS1']
        if len(ts_set) > 1:
            return None
        ts = tuple(ts_set)[0]
        ts_id = cls.convert_ts_to_id(ts)
        if ts_id == -1:
            return 'other'
        return ts

    @classmethod
    def convert_ts_to_id(cls, ts):
        ts_list = [(4, 4), (2, 4), (3, 4), (1, 4), (6, 8), (3, 8)]
        try:
            idx = ts_list.index(ts)
        except ValueError:
            idx = -1
        return idx

    def get_vector(self, use=True, use_info=None):
        # 顺序：(4, 4), (2, 4), (3, 4), (1, 4), (6, 8), (3, 8), other, NA
        value = self.value
        vector = [0] * self.vector_dim
        if not use or value is None:
            vector[-1] = 1
            return vector
        if value == 'other':
            vector[-2] = 1
            return vector
        ts_id = self.convert_ts_to_id(value)
        assert ts_id != -1
        vector[ts_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 8


class UnitTS2(UnitBase):
    """
    TS是否变化
    """

    def __init__(self, value, encoder=None):
        super().__init__(value, encoder=encoder)
        raise NotImplementedError("需要refine")

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
            Bool, True表示发生了变化，False表示没有
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
        if len(ts_set) > 1:
            return True
        return False

    def get_vector(self, use=True, use_info=None):
        value = self.value
        if value is True:
            return [1]
        else:
            return [0]

    @property
    def vector_dim(self) -> int:
        return 1
