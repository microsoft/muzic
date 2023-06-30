from .raw_unit_base import RawUnitBase


class RawUnitC1(RawUnitBase):
    """
    段落的chord序列，每小节给两个chord
    """

    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - list：段落的chord序列，每小节给两个chord
                    当MIDI的和弦因为某些问题无法检测时，返回None
        """
        if bars_chords is None:
            return None

        num_bars = len(bars_positions)
        assert num_bars * 2 == len(bars_chords)
        seg_bars_chords = bars_chords[bar_begin * 2 : bar_end * 2]

        return seg_bars_chords
