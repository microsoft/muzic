from .unit_base import UnitBase
import os

from .raw_unit_st import RawUnitST1


structure_id_to_structure_label = ["A", "AB", "AA", "ABA", "AAB", "ABB", "AAAA", "AAAB", "AABB", "AABA", "ABAA", "ABAB", "ABBA", "ABBB"]
structure_label_to_structure_id = {}
for idx, item in enumerate(structure_id_to_structure_label):
    structure_label_to_structure_id[item] = idx


def remove_digit(t):
    r = []
    for letter in t:
        if letter not in '0123456789':
            r.append(letter)
    r = ''.join(r)
    return r


def get_structure_by_file_name_1(file_name):
    assert file_name.endswith('.mid')
    file_name = file_name[:-4]
    r = file_name.split('_')[1]
    return r


st1_funcs = {
    'file_name_1': get_structure_by_file_name_1,
}


class UnitST1(UnitBase):
    """
    所有的ts种类
    """

    structure_id_to_structure_label = [
        "A", "AB", "AA", "ABA", "AAB", "ABB", "AAAA", "AAAB", "AABB", "AABA", "ABAA", "ABAB", "ABBA", "ABBB"
    ]
    structure_label_to_structure_id = {}
    for idx, item in enumerate(structure_id_to_structure_label):
        structure_label_to_structure_id[item] = idx

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitST1

    @classmethod
    def convert_raw_to_value(
        cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            str, 表示structure的字符串，包含数字
        """
        structure_label = raw_data['ST1']
        return structure_label

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * (len(self.structure_id_to_structure_label) + 1)
        if value is None or not use:
            vector[-1] = 1
            return vector

        structure = remove_digit(value)
        structure_id = self.structure_label_to_structure_id[structure]
        vector[structure_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return len(self.structure_id_to_structure_label) + 1
