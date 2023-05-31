from .unit_base import UnitBase
from .raw_unit_t import RawUnitT1


def convert_tempo_value_to_type_name_and_type_id(value):
    if value >= 200:
        return 'Prestissimo', 0
    elif value >= 168:
        return 'Presto', 1
    elif value >= 120:
        return 'Allegro', 2
    elif value >= 108:
        return 'Moderato', 3
    elif value >= 76:
        return 'Andante', 4
    elif value >= 66:
        return 'Adagio', 5
    elif value >= 60:
        return 'Larghetto', 6
    elif value >= 40:
        return 'Largo', 7
    else:
        return 'Grave', 8


class UnitT1(UnitBase):
    """
    所使用的唯一tempo
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
            - float，所使用的唯一tempo
            - str, 类别名称
            若不唯一，则两个返回值均为None
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
        if len(tempo_set) > 1:
            return None, None
        tempo = list(tempo_set)[0]
        return tempo, convert_tempo_value_to_type_name_and_type_id(tempo)[0]

    def get_vector(self, use=True, use_info=None):
        value = self.value
        tempo = value[0]
        vector = [0] * 10
        if not use or tempo is None:
            vector[-1] = 1
            return vector
        tempo_id = convert_tempo_value_to_type_name_and_type_id(tempo)[1]
        vector[tempo_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 10


class UnitT1s1(UnitBase):
    """
    演奏速度
    """
    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitT1

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """
        :return:
            - float: 所使用的唯一tempo。若有多个tempo，则返回值为None
            - int: 0表示慢，1表示适中，2表示快。若有多个tempo，则返回值为None
        """
        tempo_list = raw_data['T1']
        if len(tempo_list) > 1:
            return None, None
        tempo = tempo_list[0]
        if tempo >= 120:
            return tempo, 2
        elif tempo <= 76:
            return tempo, 0
        else:
            return tempo, 1

    def get_vector(self, use=True, use_info=None) -> list:
        # 顺序：慢，适中，快，NA
        _, label_id = self.value
        vector = [0] * 4
        if label_id is None or not use:
            vector[-1] = 1
        else:
            vector[label_id] = 1
        return vector

    @property
    def vector_dim(self):
        return 4
