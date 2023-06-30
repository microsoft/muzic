# Author: Botao Yu
import miditoolkit


def load_midi(file_path=None, file=None, midi_checker='default'):
    """
    Open and check MIDI file, return MIDI object by miditoolkit.
    :param file_path:
    :param midi_checker:
    :return:
    """
    midi_obj = miditoolkit.midi.parser.MidiFile(filename=file_path, file=file)

    if midi_checker is not None and midi_checker != 'none':
        if isinstance(midi_checker, str):
            if midi_checker == 'default':
                midi_checker = default_check_midi
            else:
                raise ValueError("midi checker does not support value: %s" % midi_checker)

        midi_checker(midi_obj)

    return midi_obj


def default_check_midi(midi_obj):
    # check abnormal values in parse result
    max_time_length = 2 ** 31
    assert all(0 <= j.start < max_time_length
               and 0 <= j.end < max_time_length
               for i in midi_obj.instruments for j in i.notes), 'Bad note time'
    assert all(0 < j.numerator < max_time_length and 0 < j.denominator < max_time_length for j in
               midi_obj.time_signature_changes), 'Bad time signature value'
    assert 0 < midi_obj.ticks_per_beat < max_time_length, 'Bad ticks per beat'

    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    assert midi_notes_count > 0, 'Blank note.'
