from .raw_unit_base import RawUnitForExistedValue


class RawUnitEM1(RawUnitForExistedValue):
    @classmethod
    def get_fields(cls):
        return 'emotion'
