from .unit_base import UnitBase
import os

from .raw_unit_em import RawUnitEM1


def get_emotion_by_file_name_1(file_name):
    assert file_name.endswith('.mid')
    file_name = file_name[:-4]
    r = file_name.split('_')[1]
    return r


em1_funcs = {
    'file_name_1': get_emotion_by_file_name_1,
}


class UnitEM1(UnitBase):
    """
    所有的ts种类
    """
    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitEM1

    @classmethod
    def convert_raw_to_value(
        cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        emo_label = raw_data['EM1']
        emo_label = emo_label['emotion']
        return emo_label

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * 5
        if value is None or not use:
            vector[-1] = 1
            return vector
        emo_id = int(value[1]) - 1
        assert 0 <= emo_id < 4
        vector[emo_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 5
