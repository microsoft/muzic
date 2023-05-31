from .unit_base import UnitBase

from .raw_unit_r import RawUnitR4, RawUnitR3


class UnitR1(UnitBase):
    """
    是否danceable
    """

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitR4

    @classmethod
    def count_on_and_off_beat_notes(cls, ts, pos_notes):
        if pos_notes is None:
            return None

        num_on = 0
        num_off = 0

        def get_on_positions(num_beat_pos, on_beat_list):
            on_position_set = list()
            for on_beat in on_beat_list:
                # assert isinstance(on_beat, int)
                # assert isinstance(num_beat_pos, int)
                for pos in range((on_beat - 1) * num_beat_pos, on_beat * num_beat_pos):
                    on_position_set.append(pos)
            on_position_set = set(on_position_set)
            return on_position_set

        ts = tuple(ts)

        if ts == (4, 4):
            beat_pos = len(pos_notes) // 4
            assert len(pos_notes) % 4 == 0
            on_beats = (1, 3)
            on_positions = get_on_positions(beat_pos, on_beats)
        elif ts == (3, 4):
            beat_pos = len(pos_notes) // 3
            assert len(pos_notes) % 3 == 0
            on_beats = (1,)
            on_positions = get_on_positions(beat_pos, on_beats)
        elif ts == (2, 4):
            beat_pos = len(pos_notes) // 2
            assert len(pos_notes) % 2 == 0
            on_beats = (1,)
            on_positions = get_on_positions(beat_pos, on_beats)
        else:
            return None

        for idx, num in enumerate(pos_notes):
            if idx in on_positions:
                num_on += 1
            else:
                num_off += 1
        return num_on, num_off

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """
        :return:
        - bool: 是否danceable，若无法判断则为None
        """
        raw_r4 = raw_data['R4']
        num_on, num_off = 0, 0
        for ts in raw_r4:
            pos_count = raw_r4[ts]
            value = cls.count_on_and_off_beat_notes(ts, pos_count)
            if value is None:
                continue
            num_on += value[0]
            num_off += value[1]
        if num_on == 0 and num_off == 0:
            return None
        if num_on > num_off:
            return True
        elif num_on < num_off:
            return False
        else:
            return None

    def get_vector(self, use=True, use_info=None) -> list:
        # 顺序：是、否、NA
        value = self.value
        vector = [0] * 3
        if value is None or not use:
            vector[2] = 1
        elif value is True:
            vector[0] = 1
        else:
            vector[1] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 3


class UnitR2(UnitBase):
    """
    是否活泼（存在某轨的跳音比例超过50%）
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
            - bool, 是否活泼。没有音符则返回None
        """

        begin = bars_positions[bar_begin][0]
        end = bars_positions[bar_end - 1][1]

        pos_resolution = encoder.vm.pos_resolution

        insts_num_notes = {}
        insts_small_dur_notes = {}

        no_notes = True
        last_tempo = None
        has_staccato = False
        for idx in range(begin, end):
            pos_item = pos_info[idx]
            tempo = pos_item[3]
            if tempo is not None:
                last_tempo = tempo
            insts_notes = pos_item[4]
            if insts_notes is None:
                continue
            for inst_id in insts_notes:
                if inst_id >= 128:
                    continue
                inst_notes = insts_notes[inst_id]
                for _, dur, _ in inst_notes:
                    no_notes = False
                    if inst_id not in insts_num_notes:
                        insts_num_notes[inst_id] = 0
                    insts_num_notes[inst_id] += 1
                    if inst_id not in insts_small_dur_notes:
                        insts_small_dur_notes[inst_id] = 0
                    num_seconds = dur * last_tempo * 60 / pos_resolution
                    if num_seconds <= 0.1:
                        insts_small_dur_notes[inst_id] += 1
                        has_staccato = True

        if no_notes:
            return None

        if not has_staccato:
            return False

        for inst_id in insts_small_dur_notes:
            num_small_notes = insts_small_dur_notes[inst_id]
            num_notes = insts_num_notes[inst_id]
            if num_small_notes / num_notes >= 0.5:
                return True
        return False

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vec = [0] * self.vector_dim
        if value is None or not use:
            vec[-1] = 1
            return vec
        if value:
            vec[0] = 1
        else:
            vec[1] = 1
        return vec

    @property
    def vector_dim(self) -> int:
        return 3


class UnitR3(UnitBase):
    """
    节奏是否激烈（note density）
    """

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitR3

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """
        :return:
            - int: 是否活泼，0为否，1为适中，2为活泼。若无法判断则为None。
        """
        if 'R3' not in raw_data:
            return None
        raw_r3 = raw_data['R3']
        if raw_r3 is None:
            return None
        if raw_r3 <= 1:
            return 0
        elif 1 < raw_r3 < 2:
            return 1
        else:
            return 2

    def get_vector(self, use=True, use_info=None) -> list:
        vector = [0] * 4
        value = self.value
        if value is None or not use:
            vector[-1] = 1
        else:
            vector[value] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        raise 4
