from .raw_unit_base import RawUnitForExistedValue


class RawUnitS1(RawUnitForExistedValue):
    @classmethod
    def get_fields(cls):
        return 'artist'


class RawUnitS2(RawUnitForExistedValue):
    @classmethod
    def get_fields(cls):
        return 'genre'
