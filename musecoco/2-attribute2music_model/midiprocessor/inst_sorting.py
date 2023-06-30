INST_TYPE_MAPPING = {
    (0, 8): 2,  # Piano
    (8, 16): 13,  # Chromatic Percussion
    (16, 24): 7,  # Organ
    (24, 32): 3,  # Guitar
    (32, 40): 1,  # Bass
    (40, 48): 5,  # String
    (48, 56): 6,  # Ensemble
    (56, 64): 9,  # Brass
    (64, 72): 8,  # Reed
    (72, 80): 10,  # Pipe
    (80, 88): 11,  # Synth Lead
    (88, 96): 12,  # Synth Pad
    (96, 104): 15,  # Synth Effect
    (104, 112): 4,  # Ethnic
    (112, 120): 14,  # Percussive
    (120, 128): 16,  # Sound Effects
    (128, 129): 0,  # Percussion
}


sort_6tracks_customized_order = {
    0: 6,  # Piano
    25: 5,  # Guitar
    32: 3,  # Bass
    43: 4,  # Double Bass
    48: 7,  # String Ensemble 1
    80: 1,  # Synth Lead 1
    128: 2,  # Percussion
}


def do_sort_insts_based_on_id(insts):
    return sorted(insts)


def do_sort_insts_based_6tracks_customization1(insts):
    return sorted(insts, key=lambda x: sort_6tracks_customized_order[x])


inst_sorting_dict = {
    'id': do_sort_insts_based_on_id,
    '6tracks_cst1': do_sort_insts_based_6tracks_customization1
}

INST_SORTING_METHODS = list(inst_sorting_dict.keys())


def get_inst_sorting_method(i):
    if i is None:
        return inst_sorting_dict['id']
    if isinstance(i, str):
        return inst_sorting_dict[i]
    return i

