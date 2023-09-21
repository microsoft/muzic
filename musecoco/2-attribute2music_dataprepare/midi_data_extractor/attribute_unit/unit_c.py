import math
from .unit_base import UnitBase
from .raw_unit_c import RawUnitC1


class UnitC1(UnitBase):
    """
    chord的变化
    """

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitC1

    @classmethod
    def is_bright(cls, chord):
        if chord == 'N.C.':
            return None
        chord = chord.split(':')
        assert len(chord) == 2
        if chord[1] in ('', 'maj7', '7'):
            return True
        return False

    @classmethod
    def convert_raw_to_value(
        cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - int：取值为0-3，表示一直明亮、一直阴暗、明亮变阴暗、阴暗变明亮。
                   取值可能为None，表示此条没有chord相关信息，或者变化的情况比较复杂，不考虑了。
        """
        seg_bars_chords = raw_data['C1']
        if seg_bars_chords is None:
            return None
        seg_brights = [cls.is_bright(item) for item in seg_bars_chords]

        num_seg_bars = bar_end - bar_begin
        num_all = len(seg_brights)
        num_brights = 0
        num_not_brights = 0
        for item in seg_brights:
            if item is True:
                num_brights += 1
            elif item is False:
                num_not_brights += 1

        if num_brights / num_all >= 0.875:
            return 0
        if num_not_brights / num_all >= 0.875:
            return 1

        break_points = set()
        for idx in range(max(1, math.floor(num_seg_bars /4)), min(num_seg_bars - 1, math.ceil(num_seg_bars * 3 / 4))):
            break_points.add(idx)
        for bp in break_points:
            num_left = bp * 2
            num_right = num_all - num_left
            num_left_bright, num_left_not_bright = 0, 0
            num_right_bright, num_right_not_bright = 0, 0
            for idx in range(num_left):
                item = seg_brights[idx]
                if item is True:
                    num_left_bright += 1
                elif item is False:
                    num_left_not_bright += 1
            for idx in range(bp * 2, num_all):
                item = seg_brights[idx]
                if item is True:
                    num_right_bright += 1
                elif item is False:
                    num_right_not_bright += 1
            if num_left_bright / num_left >= 0.875 and num_right_not_bright / num_right >= 0.875:
                return 2
            elif num_left_not_bright / num_left >= 0.875 and num_right_bright / num_right >= 0.875:
                return 3
        return None

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * self.vector_dim
        if value is None or not use:
            vector[-1] = 1
            return vector
        assert 0 <= value < 4
        vector[value] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 5
