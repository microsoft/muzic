from abc import ABC


class RawUnitBase(ABC):

    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        """
        从函数输入的内容中获取该attribute的信息，返回的信息应等于需要的信息，或者是需要信息的超集。
        你重写的函数里面，应该写清楚输出信息的格式和内容
        :param encoder: mp.MidiEncoder实例
        :param midi_dir: 数据集的全路径
        :param midi_path: MIDI相对于数据集路径的相对路径
        :param pos_info: pos_info，对于每个小节开头位置，都补齐了ts和tempo，方便使用
        :param bars_positions: dict，小节在pos_info中的开始和结束位置
        :param bars_chords: 小节序列的和弦信息，每个小节给两个bar。有可能为None，此时对于此MIDI无法抽取chord信息。
        :param bars_insts: 每个小节所用到的instrument id，列表，每个item是set
        :param bar_begin: 现在要抽取的信息的开始小节（从0开始）
        :param bar_end: 现在要抽取的信息的结束小节（不含）
        :param kwargs: 其他信息，默认为空字典
        :return:
        """
        raise NotImplementedError

    @classmethod
    def repr_value(cls, value):
        return value

    @classmethod
    def derepr_value(cls, rep_value):
        return rep_value


class RawUnitForExistedValue(RawUnitBase):
    @classmethod
    def get_fields(cls):
        raise NotImplementedError

    @classmethod
    def extract(
        cls, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs
    ):
        fields = cls.get_fields()
        if isinstance(fields, str):
            fields = (fields,)

        r = {}
        for field in fields:
            r[field] = kwargs[field] if field in kwargs else None
        return r
