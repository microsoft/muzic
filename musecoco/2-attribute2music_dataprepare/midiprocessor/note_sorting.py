def do_sort_notes_from_high_to_low(notes):
    return sorted(notes, key=lambda x: (x[0], x[1], x[2]), reverse=True)


note_sorting_dict = {
    'high_to_low': do_sort_notes_from_high_to_low,
}

NOTE_SORTING_METHODS = list(note_sorting_dict.keys())


def get_note_sorting_method(i):
    if i is None:
        return note_sorting_dict['high_to_low']
    if isinstance(i, str):
        return note_sorting_dict[i]
    return i
