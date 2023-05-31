import math

from .unit_base import UnitBase
from .raw_unit_p import RawUnitP1, RawUnitP2


class UnitP1(UnitBase):
    """
    low pitch
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
            - int，最低pitch
            没有音符则返回None
        """
        low = 1000

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        no_notes = True
        for idx in range(begin, end):
            pos_item = pos_info[idx]
            insts_notes = pos_item[4]
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]
                for pitch, _, _ in inst_notes:
                    low = min(low, pitch)
                    no_notes = False

        if no_notes:
            return None
        return low

    def get_vector(self, use=True, use_info=None):
        pitch = self.value
        vec = [0] * 129
        if pitch is None or not use:
            vec[-1] = 1
            return vec
        vec[pitch] = 1
        return vec

    @property
    def vector_dim(self) -> int:
        return 129


class UnitP2(UnitBase):
    """
    pitch range
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
            - int，最高pitch
            没有音符则返回None
        """
        high = -1

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        no_notes = True
        for idx in range(begin, end):
            pos_item = pos_info[idx]
            insts_notes = pos_item[4]
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]
                for pitch, _, _ in inst_notes:
                    high = max(high, pitch)
                    no_notes = False

        if no_notes:
            return None
        return high

    def get_vector(self, use=True, use_info=None):
        pitch = self.value
        vec = [0] * 129
        if pitch is None or not use:
            vec[-1] = 1
            return vec
        vec[pitch] = 1
        return vec

    @property
    def vector_dim(self) -> int:
        return 129


class UnitP3(UnitBase):
    """
    pitch class
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
            tuple, 包含所有pitch class
        """

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        no_notes = True
        pitch_class_set = set()
        for idx in range(begin, end):
            pos_item = pos_info[idx]
            insts_notes = pos_item[4]
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]
                for pitch, _, _ in inst_notes:
                    pitch = int(pitch)
                    pitch_class_set.add(pitch % 12)
                    no_notes = False

        if no_notes:
            return None
        return tuple(pitch_class_set)

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vec = [0] * 12
        if not use or value is None:
            return vec
        for pitch_class in value:
            vec[pitch_class] = 1
        return vec

    @property
    def vector_dim(self) -> int:
        return 12


class UnitP4(UnitBase):
    """
    pitch range
    """

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitP1, RawUnitP2

    @classmethod
    def convert_raw_to_value(
        cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - int，跨越的八度个数。没有音符则返回None
        """
        low = raw_data['P1']
        high = raw_data['P2']
        if low is None or high is None:
            return None
        return math.ceil((high - low) / 12)

    def get_vector(self, use=True, use_info=None):
        # 顺序：0个8度，1个8度，...，11个8度，NA
        value = self.value
        vec = [0] * self.vector_dim
        if value is None or not use:
            vec[-1] = 1
            return vec
        vec[value] = 1
        return vec

    @property
    def vector_dim(self) -> int:
        return 13
