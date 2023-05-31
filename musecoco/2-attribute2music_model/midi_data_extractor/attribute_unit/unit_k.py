from .unit_base import UnitBase
from .raw_unit_k import RawUnitK1


class UnitK1(UnitBase):
    """
    大调或小调
    """
    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitK1

    @classmethod
    def convert_raw_to_value(
        cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        return raw_data['K1']

    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - str：major为大调，minor为小调。可能为None，表示不知道。
        """
        is_major = None
        if 'is_major' in kwargs:
            is_major = kwargs['is_major']
        if is_major is True:
            return 'major'
        elif is_major is False:
            return 'minor'
        else:
            return None

    def get_vector(self, use=True, use_info=None):
        # 顺序：major, minor, NA
        value = self.value
        vector = [0] * self.vector_dim
        if not use or value is None:
            vector[-1] = 1
            return vector
        if value == 'major':
            vector[0] = 1
        elif value == 'minor':
            vector[1] = 1
        else:
            raise ValueError("The K1 value is \"%s\", which is abnormal." % str(value))
        return vector

    @property
    def vector_dim(self) -> int:
        return 3
