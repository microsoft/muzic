from copy import deepcopy


def convert_dict_key_to_str_(dict_data):
    keys = tuple(dict_data.keys())
    for inst_id in keys:
        dict_data[str(inst_id)] = dict_data.pop(inst_id)
    return dict_data


def convert_dict_key_to_str(dict_data):
    dict_data = deepcopy(dict_data)
    return convert_dict_key_to_str_(dict_data)


def convert_dict_key_to_int_(dict_data):
    keys = tuple(dict_data.keys())
    for inst_id in keys:
        dict_data[int(inst_id)] = dict_data.pop(inst_id)
    return dict_data


def convert_dict_key_to_int(dict_data):
    dict_data = deepcopy(dict_data)
    return convert_dict_key_to_int_(dict_data)


def convert_dict_key_with_eval_(dict_data):
    keys = tuple(dict_data.keys())
    for inst_id in keys:
        dict_data[eval(inst_id)] = dict_data.pop(inst_id)
    return dict_data


def convert_dict_key_with_eval(dict_data):
    dict_data = deepcopy(dict_data)
    return convert_dict_key_with_eval_(dict_data)
