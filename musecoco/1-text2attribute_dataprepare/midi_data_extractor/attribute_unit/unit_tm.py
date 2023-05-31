from .unit_base import UnitBase

from .raw_unit_tm import RawUnitTM1


class UnitTM1(UnitBase):
    """
    片段时长
    """

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitTM1

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """
        :return:
            - float: 片段时长，单位为秒
            - int: 分段id，0表示0-15秒，1表示15-30秒，2表示30-45秒，3表示45-60秒，4表示60秒以上，均为左开右闭区间。
        """
        time_second = raw_data['TM1']
        if 0 < time_second <= 15:
            return time_second, 0
        elif 15 < time_second <= 30:
            return time_second, 1
        elif 30 < time_second <= 45:
            return time_second, 2
        elif 45 < time_second <= 60:
            return time_second, 3
        else:
            return time_second, 4

    def get_vector(self, use=True, use_info=None) -> list:
        # 顺序：区间0、1、2、3、4、NA
        _, label_id = self.value
        vector = [0] * 6
        if not use:
            vector[-1] = 1
        else:
            vector[label_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 6
