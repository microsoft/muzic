from .magenta_chord_recognition import _PITCH_CLASS_NAMES, _CHORD_KIND_PITCHES, NO_CHORD


_PITCH_CLASS_NAMES_TO_INDEX = dict()
for idx, pitch_class_name in enumerate(_PITCH_CLASS_NAMES):
    _PITCH_CLASS_NAMES_TO_INDEX[pitch_class_name] = idx


def get_chord_pitch_indices(chord_label):
    if chord_label == NO_CHORD:
        return []
    pitch_class, chord_kind = chord_label.split(':')
    root_index = _PITCH_CLASS_NAMES_TO_INDEX[pitch_class]
    return [root_index + offset for offset in _CHORD_KIND_PITCHES[chord_kind]]


def convert_pitch_index_to_token(pitch_index, offset=0, min_pitch=None, max_pitch=None, tag='p'):
    p = pitch_index + offset
    if min_pitch is not None:
        while p < min_pitch:
            p += 12
    if max_pitch is not None:
        while p > max_pitch:
            p -= 12
    assert p >= 0, (pitch_index, offset, min_pitch, max_pitch)
    if min_pitch is not None:
        assert p >= min_pitch, (pitch_index, offset, min_pitch, max_pitch)
    if max_pitch is not None:
        assert p <= max_pitch, (pitch_index, offset, min_pitch, max_pitch)
    return '%s-%d' % (tag, p)


def get_chord_pitch_tokens(chord_label, offset=0, min_pitch=None, max_pitch=None, tag='p'):
    pitch_indices = get_chord_pitch_indices(chord_label)
    return [
        convert_pitch_index_to_token(
            item, offset=offset, min_pitch=min_pitch, max_pitch=max_pitch, tag=tag
        ) for item in pitch_indices
    ]
