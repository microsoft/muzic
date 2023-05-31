import importlib
from .unit_base import UnitBase


def load_unit_class(attribute_label):
    unit_file_label = []
    for letter in attribute_label:
        if letter in '0123456789':
            break
        else:
            unit_file_label.append(letter.lower())
    unit_file_label = ''.join(unit_file_label)
    module = importlib.import_module(
        '.attribute_unit.unit_%s' % unit_file_label, package='midi_data_extractor'
    )
    unit_cls = getattr(module, 'Unit%s' % attribute_label)
    return unit_cls


def load_raw_unit_class(raw_attribute_label):
    unit_file_label = []
    for letter in raw_attribute_label:
        if letter in '0123456789':
            break
        else:
            unit_file_label.append(letter.lower())
    unit_file_label = ''.join(unit_file_label)
    module = importlib.import_module(
        '.attribute_unit.raw_unit_%s' % unit_file_label, package='midi_data_extractor'
    )
    unit_cls = getattr(module, 'RawUnit%s' % raw_attribute_label)
    return unit_cls


def convert_value_into_unit(attribute_label, attribute_value, encoder=None):
    unit_cls = load_unit_class(attribute_label)
    unit = unit_cls(attribute_value, encoder=encoder)
    return unit


def convert_value_dict_into_unit_dict(value_dict, encoder=None):
    unit_dict = {}
    for attr_label in value_dict:
        unit_dict[attr_label] = convert_value_into_unit(attr_label, value_dict[attr_label], encoder=encoder)
    return unit_dict
