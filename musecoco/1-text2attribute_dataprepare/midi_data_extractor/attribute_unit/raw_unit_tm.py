from .raw_unit_base import RawUnitBase


class RawUnitTM1(RawUnitBase):
    """
    片段时长
    """

    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - float, 时长，单位为分。无None情况
        """

        pos_resolution = 12
        assert pos_resolution == encoder.vm.pos_resolution, str(encoder.vm.pos_resolution)
        pos_dict = {}

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]
        assert pos_info[begin][3] is not None
        last_tempo = None
        for idx in range(begin, end):
            tempo = pos_info[idx][3]
            if tempo is not None:
                last_tempo = tempo
                if last_tempo not in pos_dict:
                    pos_dict[last_tempo] = 0
            pos_dict[last_tempo] += 1

        time_second = 0
        for tempo in pos_dict:
            n = pos_dict[tempo] * 60 / pos_resolution / tempo
            time_second += n

        return time_second
