from .raw_unit_base import RawUnitBase


class RawUnitK1(RawUnitBase):
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        大调或小调
        :return:
            - str: major为大调，minor为小调。可能为None，表示不知道。
        """
        r = None
        if 'is_major' in kwargs:
            is_major = kwargs['is_major']
            if is_major is True:
                r = 'major'
            elif is_major is False:
                r = 'minor'
            elif is_major is None:
                r = None
            else:
                raise ValueError('is_major argument is set to a wrong value:', is_major)
        return r
