from .unit_base import UnitBase
import os

from .raw_unit_s import RawUnitS1, RawUnitS2


def s1_func_by_is_symphony(file_path):
    return True


def s1_func_by_has_symphony_1(file_path):
    file_path = file_path.replace('\\', '/')
    if 'symphony' in file_path:
        return True
    return None


s1_funcs = {
    'is_symphony': s1_func_by_is_symphony,
    'has_symphony_1': s1_func_by_has_symphony_1
}


class UnitS1(UnitBase):
    """
    是否是交响乐
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
            - bool，表示是否是交响乐
            可能为None，表示不知道是否为交响乐
        """
        if 's1_func' not in kwargs:
            return None
        judge_func = kwargs['s1_func']
        if judge_func is None:
            return None

        judge_func = s1_funcs[kwargs['s1_func']]
        file_name = os.path.basename(midi_path)
        is_symphony = judge_func(file_name)
        return is_symphony

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * self.vector_dim
        if value is None or not use:
            vector[-1] = 1
            return vector
        if value:
            vector[0] = 1
        else:
            vector[1] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 3


dir_name_to_artist_name = {
    'beethoven': 'Beethoven',
    'mozart': 'Mozart',
    'chopin': 'Chopin',
    'schubert': 'Schubert',
    'schumann': 'Schumann',
}

artist_name_to_id = {
    'Beethoven': 0,
    'Mozart': 1,
    'Chopin': 2,
    'Schubert': 3,
    'Schumann': 4,
}


def s2_func_by_file_path_1(file_path):
    file_path = file_path.replace('\\', '/')
    file_path_split = file_path.split('/')
    first_dir = file_path_split[0]
    if first_dir in dir_name_to_artist_name:
        return dir_name_to_artist_name[first_dir]
    return None


s2_funcs = {
    'file_path_1': s2_func_by_file_path_1,
}


class UnitS2(UnitBase):
    """
    是否是某艺术家的作品
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
            - str，艺术家名字
            可能为None，表示不知道艺术家是谁
        """
        if 's2_func' not in kwargs:
            return None
        judge_func = kwargs['s2_func']
        if judge_func is None:
            return None

        judge_func = s2_funcs[kwargs['s2_func']]
        artist_name = judge_func(midi_path)
        return artist_name

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * self.vector_dim
        if value is None or not use:
            vector[-1] = 1
            return vector
        value_id = artist_name_to_id[value]
        assert 0 <= value_id < self.vector_dim - 1
        vector[value_id] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return len(artist_name_to_id) + 1


class UnitS2s1(UnitBase):
    """
    艺术家
    """

    artist_label_to_artist_id = {
        'beethoven': 0,
        'mozart': 1,
        'chopin': 2,
        'schubert': 3,
        'schumann': 4,
        'bach-js': 5,
        'haydn': 6,
        'brahms': 7,
        'Handel': 8,
        'tchaikovsky': 9,
        'mendelssohn': 10,
        'dvorak': 11,
        'liszt': 12,
        'stravinsky': 13,
        'mahler': 14,
        'prokofiev': 15,
        'shostakovich': 16,
    }

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitS1

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """
        :return:
        - str，艺术家label。可能为None，表示不知道艺术家是谁
        """
        raw_s1 = raw_data['S1']
        raw_s1 = raw_s1['artist']
        return raw_s1

    @classmethod
    def convert_label_to_id(cls, label):
        return cls.artist_label_to_artist_id[label]

    def get_vector(self, use=True, use_info=None) -> list:
        # 顺序：artist 0, artist 1, ..., NA
        vector = [0] * (len(self.artist_label_to_artist_id) + 1)
        if not use or self.value is None:
            vector[-1] = 1
        else:
            label_id = self.convert_label_to_id(self.value)
            vector[label_id] = 1
        return vector

    @property
    def vector_dim(self):
        return len(self.artist_label_to_artist_id) + 1


def s3_func_by_is_classical(file_name):
    return True


def s3_func_by_has_classical_1(file_path):
    file_path = file_path.replace('\\', '/')
    if 'classical' in file_path:
        return True
    return None


s3_funcs = {
    'is_classical': s3_func_by_is_classical,
    'has_classical_1': s3_func_by_has_classical_1,
}


class UnitS3(UnitBase):
    """
    是否是古典乐
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
            - bool，表示是否是古典乐
            可能为None，表示不知道是否为古典乐
        """
        if 's3_func' not in kwargs:
            return None
        judge_func = kwargs['s3_func']
        if judge_func is None:
            return None

        judge_func = s3_funcs[kwargs['s3_func']]
        is_classical = judge_func(midi_path)
        return is_classical

    def get_vector(self, use=True, use_info=None):
        value = self.value
        vector = [0] * self.vector_dim
        if value is None or not use:
            vector[-1] = 1
            return vector
        if value:
            vector[0] = 1
        else:
            vector[1] = 1
        return vector

    @property
    def vector_dim(self) -> int:
        return 3


class UnitS4(UnitBase):
    """
    Genre
    """
    genre_label_to_genre_id = {
        'New Age': 0,
        'Electronic': 1,
        'Rap': 2,
        'Religious': 3,
        'International': 4,
        'Easy_Listening': 5,
        'Avant_Garde': 6,
        'RnB': 7,
        'Latin': 8,
        'Children': 9,
        'Jazz': 10,
        'Classical': 11,
        'Comedy_Spoken': 12,
        'Pop_Rock': 13,
        'Reggae': 14,
        'Stage': 15,
        'Folk': 16,
        'Blues': 17,
        'Vocal': 18,
        'Holiday': 19,
        'Country': 20,
        'Symphony': 21,
    }

    @classmethod
    def convert_label_to_id(cls, label):
        return cls.genre_label_to_genre_id[label]

    @classmethod
    def get_raw_unit_class(cls):
        return RawUnitS2

    @classmethod
    def convert_raw_to_value(cls, raw_data, encoder, midi_dir, midi_path,
        pos_info, bars_positions, bars_chords, bars_insts, bar_begin, bar_end, **kwargs):
        """
        :return:
        - tuple of str: 所有适用的genre label，已去重。若不知道则为None。
        """
        raw_s2 = raw_data['S2']
        raw_s2 = raw_s2['genre']
        raw_s2 = tuple(set(raw_s2)) if raw_s2 is not None else None
        return raw_s2

    def get_vector(self, use=True, use_info=None):
        # 返回genre种数个列表，每个列表顺序：是, 否, NA
        vector = [[0, 0, 0] for _ in range(len(self.genre_label_to_genre_id))]
        if not use:
            for item in vector:
                item[2] = 1
            return vector
        if use_info is not None:
            used_genres, unused_genres = use_info
            usedNone = True
            unusedNone = True
            if used_genres != None:
                used_genres = set(used_genres)
                usedNone = False
            else:
                used_genres = set()
            if unused_genres != None:
                unused_genres = set(unused_genres)
                unusedNone = False
            else:
                unused_genres = set()
            if usedNone == False and unusedNone == False:
                assert len(used_genres & unused_genres) == 0
            if usedNone == False:
                for genre in used_genres:
                    genre_id = self.convert_label_to_id(genre)
                    vector[genre_id][0] = 1
            if unusedNone == False:
                for genre in unused_genres:
                    genre_id = self.convert_label_to_id(genre)
                    vector[genre_id][1] = 1
            na_insts = set(self.genre_label_to_genre_id.keys()) - used_genres - unused_genres
            for genre in na_insts:
                genre_id = self.convert_label_to_id(genre)
                vector[genre_id][2] = 1
        else:
            value = self.value
            if value is None:
                value = tuple()
            for genre in value:
                genre_id = self.convert_label_to_id(genre)
                vector[genre_id][0] = 1
            na_insts = set(self.genre_label_to_genre_id.keys()) - set(value)
            for genre in na_insts:
                genre_id = self.convert_label_to_id(genre)
                vector[genre_id][2] = 1
        return vector

    @property
    def vector_dim(self):
        return len(self.genre_label_to_genre_id), 3
