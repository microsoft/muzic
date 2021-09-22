import math
import miditoolkit
import numpy as np
from utils.midi_io import midi2items, TICKS_PER_STEP, Item
from utils.chord_recog import infer_chords_for_sequence, NO_CHORD


def infer_chords_for_midi(midi_file, instru2track=None):
    note_items, tempo = midi2items(midi_file, instru2track=instru2track, before_infer_chords=True)
    ticks_per_beat = midi_file.ticks_per_beat
    required_instru = [[2, 3], [5]]
    max_beats = math.ceil(midi_file.max_tick / ticks_per_beat)
    note_per_beats = np.zeros([len(required_instru), max_beats])
    note_per_2beats = np.zeros([len(required_instru), math.ceil(max_beats / 2)])
    notes_no_drum = []
    for idx, instru_idx in enumerate(required_instru):  # piano guitar string
        notes_no_drum.append([n for n in note_items if n.track in instru_idx])
        if len(notes_no_drum[idx]) > 0:
            for note in notes_no_drum[idx]:
                start_beat = note.start // ticks_per_beat
                end_beat = (note.end - 1) // ticks_per_beat
                for i in range(start_beat, end_beat + 1):
                    note_per_beats[idx][i] += 1
                for i in range(start_beat // 2, end_beat // 2 + 1):
                    note_per_2beats[idx][i] += 1
    chord_items = [Item(
        name="Chord",
        start=b * ticks_per_beat,
        end=b * ticks_per_beat,
        value=NO_CHORD
    ) for b in range(max_beats)]
    for idx in range(len(required_instru)):
        break_flag = True
        if len(notes_no_drum[idx]) > 0:
            _, this_chord_items = infer_chords_for_sequence(notes_no_drum[idx], tempo, ticks_per_beat, chords_per_bar=4)
            for i in range(min(max_beats, len(this_chord_items))):
                if chord_items[i].value == NO_CHORD:
                    chord_items[i] = this_chord_items[i]
                    if note_per_beats[idx][i] < 3:
                        chord_items[i].value = NO_CHORD
                    break_flag = False
            if break_flag:
                break
            _, this_chord_items = infer_chords_for_sequence(notes_no_drum[idx], tempo, ticks_per_beat, chords_per_bar=2)
            for i in range(min(max_beats, len(this_chord_items) * 2)):
                if chord_items[i].value == NO_CHORD:
                    chord_items[i].value = this_chord_items[i // 2].value
                    if note_per_2beats[idx][i // 2] < 3:
                        chord_items[i].value = NO_CHORD
                    break_flag = False
            if break_flag:
                break
    # print(chord_items)
    chords = []
    for chord in chord_items:
        # if chord.value == NO_CHORD:
        #     continue
        chords.append([chord.start, chord.value])
    if len(chords) > 0:
        if len(midi_file.markers):
            midi_file.markers = []
        for c in chords:
            midi_file.markers.append(miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    return midi_file


def midi2chords(midi_fn, instru2track):
    mf = miditoolkit.MidiFile(midi_fn)
    midi = infer_chords_for_midi(mf, instru2track=instru2track)
    for c in midi.markers:
        c.time = int(round(c.time / TICKS_PER_STEP))
    return [[x.time, x.text] for x in midi.markers]