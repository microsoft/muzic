from .unit_base import UnitBase


class UnitM1(UnitBase):
    """
    track repetition
    """

    @classmethod
    def get_raw_unit_class(cls):
        raise NotImplementedError

    @classmethod
    def convert_raw_to_value(cls, raw_data):
        """
        :return:
            - dict: key为inst_id（原始id，0-128），value为bool，表示是否有重复，若value为None则表示无法判断。
        """
        pass

    @property
    def vector_dim(self) -> int:
        raise NotImplementedError

    def get_vector(self, use=True, use_info=None) -> list:
        raise NotImplementedError


class UnitM2(UnitBase):
    """
    melody pattern
    """

    @classmethod
    def get_raw_unit_class(cls):
        raise NotImplementedError

    @classmethod
    def convert_raw_to_value(cls, raw_data):
        """
        :return:
            - int: 0表示上升，1表示下降，2表示上升后下降，3表示下降后上升，4表示平。None表示不予考虑的其他情况
        """
        pass

    @property
    def vector_dim(self) -> int:
        raise NotImplementedError

    def get_vector(self, use=True, use_info=None) -> list:
        raise NotImplementedError
