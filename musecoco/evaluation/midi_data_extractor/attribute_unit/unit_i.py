from typing import Tuple, Union

from .unit_base import UnitBase
from ..const import inst_id_to_inst_class_id, inst_id_to_inst_class_id_2

from .raw_unit_i import RawUnitI1
from .raw_unit_p import RawUnitP3
from .raw_unit_n import RawUnitN2


class UnitI1(UnitBase):
    """
    所用的乐器（大类）
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
            tuple，包含所有乐器的大类id，已去重，若空则返回None
        """
        all_insts = set()
        for bar_insts in bars_insts[bar_begin: bar_end]:
            all_insts = all_insts | bar_insts
        all_inst_classes = []
        # print(all_insts)
        # print(bars_insts[bar_begin: bar_end])
        for inst_id in all_insts:
            all_inst_classes.append(inst_id_to_inst_class_id[inst_id])
        if len(all_inst_classes) == 0:
            return None
        return tuple(set(all_inst_classes))

    def get_vector(self, use=True, use_info=None):
        vector = [0] * 17
        if use_info is None:
            value = self.value
        else:
            value = use_info
        if not use or value is None:
            return vector
        for inst_class_id in value:
            vector[inst_class_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 17


class UnitI1s1(UnitBase):
    """
    所用的乐器（大类v2）
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
            tuple，包含所有乐器的大类id，已去重，若空则返回None
        """
        all_insts = set()
        for bar_insts in bars_insts[bar_begin: bar_end]:
            all_insts = all_insts | bar_insts
        all_inst_classes = []
        # print(all_insts)
        # print(bars_insts[bar_begin: bar_end])
        for inst_id in all_insts:
            all_inst_classes.append(inst_id_to_inst_class_id_2[inst_id])
        if len(all_inst_classes) == 0:
            return None
        return tuple(set(all_inst_classes))

    def get_vector(self, use=True, use_info=None):
        vector = [0] * self.vector_dim
        if use_info is None:
            value = self.value
        else:
            value = use_info
        if not use or value is None:
            return vector
        for inst_class_id in value:
            vector[inst_class_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 14


class UnitI1s2(UnitBase):
    """
    所用的乐器（大类v3）
    """
    inst_class_version = 'v3'

    inst_id_to_inst_class_id = {
        # piano 0:
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,

        # keyboard 1:
        6: 1,
        7: 1,
        8: 1,
        9: 1,

        # percussion 2:
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        47: 2,
        55: 2,
        112: 2,
        113: 2,
        115: 2,
        117: 2,
        119: 2,

        # organ 3:
        16: 3,
        17: 3,
        18: 3,
        19: 3,
        20: 3,
        21: 3,
        22: 3,
        23: 3,

        # guitar 4:
        24: 4,
        25: 4,
        26: 4,
        27: 4,
        28: 4,
        29: 4,
        30: 4,
        31: 4,

        # bass 5:
        32: 5,
        33: 5,
        34: 5,
        35: 5,
        36: 5,
        37: 5,
        38: 5,
        39: 5,
        43: 5,

        # violin 6:
        40: 6,

        # viola 7:
        41: 7,

        # cello 8:
        42: 8,

        # harp 9:
        46: 9,

        # strings 10:
        44: 10,
        45: 10,
        48: 10,
        49: 10,
        50: 10,
        51: 10,

        # voice 11:
        52: 11,
        53: 11,
        54: 11,

        # trumpet 12:
        56: 12,
        59: 12,

        # trombone 13:
        57: 13,

        # tuba 14:
        58: 14,

        # horn 15:
        60: 15,
        69: 15,

        # brass 16:
        61: 16,
        62: 16,
        63: 16,

        # sax 17:
        64: 17,
        65: 17,
        66: 17,
        67: 17,

        # oboe 18:
        68: 18,

        # bassoon 19:
        70: 19,

        # clarinet 20:
        71: 20,

        # piccolo 21:
        72: 21,

        # flute 22:
        73: 22,
        75: 22,

        # pipe 23:
        74: 23,
        76: 23,
        77: 23,
        78: 23,
        79: 23,

        # synthesizer 24:
        80: 24,
        81: 24,
        82: 24,
        83: 24,
        84: 24,
        85: 24,
        86: 24,
        87: 24,
        88: 24,
        89: 24,
        90: 24,
        91: 24,
        92: 24,
        93: 24,
        94: 24,
        95: 24,

        # ethnic instrument 25:
        104: 25,
        105: 25,
        106: 25,
        107: 25,
        108: 25,
        109: 25,
        110: 25,
        111: 25,

        # sound effect 26:
        10: 26,
        120: 26,
        121: 26,
        122: 26,
        123: 26,
        124: 26,
        125: 26,
        126: 26,
        127: 26,
        96: 26,
        97: 26,
        98: 26,
        99: 26,
        100: 26,
        101: 26,
        102: 26,
        103: 26,

        # drum 27:
        128: 27,
        118: 27,
        114: 27,
        116: 27,
    }

    inst_class_id_to_inst_class_name = {
        # piano 0:
        0: 'piano',

        # keyboard 1:
        1: 'keyboard',

        # percussion 2:
        2: 'percussion',

        # organ 3:
        3: 'organ',

        # guitar 4:
        4: 'guitar',

        # bass 5:
        5: 'bass',

        # violin 6:
        6: 'violin',

        # viola 7:
        7: 'viola',

        # cello 8:
        8: 'cello',

        # harp 9:
        9: 'harp',

        # strings 10:
        10: 'strings',

        # voice 11:
        11: 'voice',

        # trumpet 12:
        12: 'trumpet',

        # trombone 13:
        13: 'trombone',

        # tuba 14:
        14: 'tuba',

        # horn 15:
        15: 'horn',

        # brass 16:
        16: 'brass',

        # sax 17:
        17: 'sax',

        # oboe 18:
        18: 'oboe',

        # bassoon 19:
        19: 'bassoon',

        # clarinet 20:
        20: 'clarinet',

        # piccolo 21:
        21: 'piccolo',

        # flute 22:
        22: 'flute',

        # pipe 23:
        23: 'pipe',

        # synthesizer 24:
        24: 'synthesizer',

        # ethnic instrument 25:
        25: 'ethnic instrument',

        # sound effect 26:
        26: 'sound effect',

        # drum 27:
        27: 'drum',
    }

    inst_class_name_to_inst_class_id = {}
    for inst_class_id in inst_class_id_to_inst_class_name:
        inst_class_name = inst_class_id_to_inst_class_name[inst_class_id]
        inst_class_name_to_inst_class_id[inst_class_name] = inst_class_id

    num_classes = len(inst_class_id_to_inst_class_name)

    @classmethod
    def convert_inst_id_to_inst_class_id(cls, inst_id):
        return cls.inst_id_to_inst_class_id[inst_id]

    @classmethod
    def convert_inst_class_id_to_inst_class_name(cls, inst_class_id):
        return cls.inst_class_id_to_inst_class_name[inst_class_id]

    @classmethod
    def convert_inst_class_name_to_inst_class_id(cls, inst_class_name):
        return cls.inst_class_name_to_inst_class_id[inst_class_name]

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitI1

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """
        :return:
            - tuple，包含所有乐器的大类id，已去重，若空则返回None
        """
        r = raw_data['I1']
        if len(r) == 0:
            return None
        nr = set()
        for inst_id in r:
            nr.add(cls.convert_inst_id_to_inst_class_id(inst_id))
        nr = tuple(nr)
        return nr

    def get_vector(self, use=True, use_info=None) -> list:
        "乐器个列表，每个列表长度为3，依次为是、否、NA"
        value = self.value  # tuple
        vector = [[0, 0, 0] for _ in range(len(self.inst_class_id_to_inst_class_name))]
        if not use:
            for item in vector:
                item[2] = 1
            return vector
        if use_info is not None:
            used_insts, unused_insts = use_info
            usedNone = True
            unusedNone = True
            if used_insts != None:
                used_insts = set(used_insts)
                usedNone = False
            else:
                used_insts = set()
            if unused_insts != None:
                unused_insts = set(unused_insts)
                unusedNone = False
            else:
                unused_insts = set()
            if unusedNone == False and usedNone == False:
                assert len(used_insts & unused_insts) == 0
            if usedNone==False:
                for inst_class_id in used_insts:
                    vector[inst_class_id][0] = 1
            if unusedNone == False:
                for inst_class_id in unused_insts:
                    vector[inst_class_id][1] = 1
            na_insts = set(range(len(self.inst_class_id_to_inst_class_name))) - used_insts - unused_insts
            for inst_class_id in na_insts:
                vector[inst_class_id][2] = 1
        else:
            if value is None:
                value = tuple()
            for inst_class_id in value:
                vector[inst_class_id][0] = 1
            na_insts = set(range(len(self.inst_class_id_to_inst_class_name))) - set(value)
            for inst_class_id in na_insts:
                vector[inst_class_id][2] = 1
        return vector

    @property
    def vector_dim(self) -> Tuple[int, int]:
        return len(self.inst_class_id_to_inst_class_name), 3


class UnitI2(UnitBase):
    """
    乐器（大类）的增加或减少
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
            若返回一个东西，则必为None，表示没有可用的乐器增减情况，此条样本可忽略。
            若返回四个东西，则：
                第一个值为'inc'或'dec'，表示有乐器增或减
                第二个值为set，包含增的乐器class id
                第三个值为set，包含减的乐器class id
                第四个值为int，表示增减的bar的索引
        """
        seg_bars_insts = bars_insts[bar_begin: bar_end]
        temp = []
        for bar_insts in seg_bars_insts:
            temp_set = set()
            for inst_id in bar_insts:
                temp_set.add(inst_id_to_inst_class_id[inst_id])
            temp.append(temp_set)
        seg_bars_insts = temp
        last_insts = []
        change_point = None
        for idx, bar_insts in enumerate(seg_bars_insts):
            if len(last_insts) == 0:
                last_insts.append(bar_insts)
            else:
                if last_insts[-1] != bar_insts:
                    last_insts.append(bar_insts)
                    change_point = idx
            if len(last_insts) > 2:
                return None
        if len(last_insts) != 2:
            return None
        increased_insts = tuple(last_insts[1] - last_insts[0])
        decreased_insts = tuple(last_insts[0] - last_insts[1])
        if len(increased_insts) > 0 and len(decreased_insts) == 0:
            return 'inc', increased_insts, None, change_point
        elif len(increased_insts) == 0 and len(decreased_insts) > 0:
            return 'dec', None, decreased_insts, change_point
        else:
            return None

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * 34
        if value is None or not use:
            return vector
        change_type, inc_insts, dec_insts, change_point = value
        offset = 0 if change_type == 'inc' else 17
        change_insts = inc_insts if change_type == 'inc' else dec_insts
        for inst_class_id in change_insts:
            vector[inst_class_id + offset] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 34


class UnitI2s1(UnitBase):
    """
    乐器（大类v2）的增加或减少
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
            若返回一个东西，则必为None，表示没有可用的乐器增减情况，此条样本可忽略。
            若返回四个东西，则：
                第一个值为'inc'或'dec'，表示有乐器增或减
                第二个值为set，包含增的乐器class id
                第三个值为set，包含减的乐器class id
                第四个值为int，表示增减的bar的索引
        """
        seg_bars_insts = bars_insts[bar_begin: bar_end]
        temp = []
        for bar_insts in seg_bars_insts:
            temp_set = set()
            for inst_id in bar_insts:
                temp_set.add(inst_id_to_inst_class_id_2[inst_id])
            temp.append(temp_set)
        seg_bars_insts = temp
        last_insts = []
        change_point = None
        for idx, bar_insts in enumerate(seg_bars_insts):
            if len(last_insts) == 0:
                last_insts.append(bar_insts)
            else:
                if last_insts[-1] != bar_insts:
                    last_insts.append(bar_insts)
                    change_point = idx
            if len(last_insts) > 2:
                return None
        if len(last_insts) != 2:
            return None
        increased_insts = tuple(last_insts[1] - last_insts[0])
        decreased_insts = tuple(last_insts[0] - last_insts[1])
        if len(increased_insts) > 0 and len(decreased_insts) == 0:
            return 'inc', increased_insts, None, change_point
        elif len(increased_insts) == 0 and len(decreased_insts) > 0:
            return 'dec', None, decreased_insts, change_point
        else:
            return None

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * self.vector_dim
        if value is None or not use:
            return vector
        change_type, inc_insts, dec_insts, change_point = value
        offset = 0 if change_type == 'inc' else 14
        change_insts = inc_insts if change_type == 'inc' else dec_insts
        for inst_class_id in change_insts:
            vector[inst_class_id + offset] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 28


# class UnitI3(UnitBase):
#     """
#     前半段和后半段的乐器（大类v3）的变化
#     """
#
#     @classmethod
#     def convert_raw_to_value(cls, raw_data):
#         """
#
#         :return:
#         - tuple，后半段相对于前半段增加的乐器的大类id，已去重，若空则返回None
#         - tuple，后半段相对于前半段减少的乐器的大类id，已去重，若空则返回None
#         """
#         pass


class UnitI4(UnitBase):
    """
    演奏旋律的乐器（大类v3）
    """

    inst_class_version = 'v3'

    inst_id_to_inst_class_id = UnitI1s2.inst_id_to_inst_class_id

    inst_class_id_to_inst_class_name = UnitI1s2.inst_class_id_to_inst_class_name

    inst_class_name_to_inst_class_id = UnitI1s2.inst_class_name_to_inst_class_id

    num_classes = UnitI1s2.num_classes

    @classmethod
    def convert_inst_id_to_inst_class_id(cls, inst_id):
        return cls.inst_id_to_inst_class_id[inst_id]

    @classmethod
    def convert_inst_class_id_to_inst_class_name(cls, inst_class_id):
        return cls.inst_class_id_to_inst_class_name[inst_class_id]

    @classmethod
    def convert_inst_class_name_to_inst_class_id(cls, inst_class_name):
        return cls.inst_class_name_to_inst_class_id[inst_class_name]

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitP3, RawUnitN2

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """

        :return:
        - int: 演奏旋律的乐器大类（v3）id。若无法检测到旋律乐器，则返回None。
        - bool: 若只有一个非鼓（原始类128）乐器，返回True，否则返回False
        """
        raw_p3 = raw_data['P3']
        raw_n2 = raw_data['N2']

        if len(raw_p3) == 0:
            r = None
        else:
            avg_pitch_dict = {}
            for inst_id in raw_p3:
                avg_pitch_dict[inst_id] = raw_p3[inst_id] / raw_n2[inst_id]
            sorted_list = sorted(avg_pitch_dict.items(), key=lambda x: x[0], reverse=True)
            candidate_inst_id = sorted_list[0][0]
            if raw_n2[candidate_inst_id] > 20:
                r = candidate_inst_id
                r = cls.convert_inst_id_to_inst_class_id(r)
            else:
                r = None

        if len(raw_p3) == 1:
            sin = True
        else:
            sin = False
        return r, sin

    def get_vector(self, use=True, use_info=None) -> list:
        value = self.value
        r, sin = value
        vector = [0] * (self.num_classes + 1)
        if not use or r is None:
            vector[-1] = 1
        else:
            vector[r] = 1
        return vector

    def vector_dim(self) -> Union[int, Tuple[int, int]]:
        return self.num_classes + 1
