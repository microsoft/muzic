from .raw_unit_base import RawUnitBase


class RawUnitM1(RawUnitBase):
    """
    各轨的SSM
    """
    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        :return:
            - dict: key是inst_id
                    value是dict, key为(i, j)表示bar i和bar j（仅包含j < i的情况），value为两bar之间的相似性
        """
        ssm = kwargs['ssm']
        r = {}
        for inst_id in ssm:
            r[inst_id] = ssm[bar_begin: bar_end, bar_begin: bar_end]
        return r
