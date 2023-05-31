from .raw_unit_base import RawUnitForExistedValue


class RawUnitST1(RawUnitForExistedValue):
    @classmethod
    def get_fields(cls):
        return 'piece_structure'
