from collections import Counter
import miditoolkit
import numpy as np


def keep_long_notes(mf, new_program_id, track_to_merge, name):
    is_drums = name == 'Drums'
    if is_drums:
        new_program_id = 0
    new_instr = miditoolkit.Instrument(new_program_id, is_drums, name)
    for t in track_to_merge:
        new_instr.notes += mf.instruments[t].notes
    new_instr.notes.sort(key=lambda x: (x.pitch, x.start, -x.end, -x.velocity))
    new_notes = [new_instr.notes[0]]
    for i in range(1, len(new_instr.notes)):
        n0 = new_notes[-1]
        n1 = new_instr.notes[i]
        if n1.start >= n0.end or n1.pitch != n0.pitch:
            new_notes.append(n1)
        elif n1.start > n0.start:
            n0.end = n1.start
            new_notes.append(n1)
    new_instr.notes = new_notes
    new_instr.notes.sort(key=lambda x: (x.start, x.pitch, -x.end))
    return new_instr


def keep_track_with_most_notes(mf, new_program_id, track_to_merge, name):
    is_drums = name == 'Drums'
    if is_drums:
        new_program_id = 0
    new_instrs = []
    for t in track_to_merge:
        new_instr = miditoolkit.Instrument(new_program_id, is_drums, name)
        new_instr.notes += mf.instruments[t].notes
        new_instrs.append(new_instr)
    new_instrs.sort(key=lambda track: len(track.notes))
    return new_instrs[-1]


def merge_lead(mf, new_program_id, track_to_merge):
    new_instr = keep_long_notes(mf, new_program_id, track_to_merge, 'Lead')
    # 多音删除
    note_start_dict = {}
    for note in new_instr.notes:
        if note.start not in note_start_dict.keys() or note.pitch > note_start_dict[note.start].pitch:
            note_start_dict[note.start] = note
    new_instr.notes = list(note_start_dict.values())
    new_instr.notes.sort(key=lambda x: (x.start, x.pitch, -x.end))
    return new_instr


def merge_strings(mf, new_program_id, track_to_merge):
    mono_tracks = []
    poly_tracks = []
    for t in track_to_merge:
        n_start = []
        pitches = []
        new_instr = miditoolkit.Instrument(new_program_id, False, 'Strings')
        for n in mf.instruments[t].notes:
            n_start.append(n.start)
            pitches.append(n.pitch)
        c_n = Counter(n_start)
        if sum(c_n.values()) / len(c_n.keys()) >= 1.2:
            new_instr.notes += mf.instruments[t].notes
            poly_tracks.append(new_instr)
        else:
            mono_tracks.append([t, pitches])
    if len(poly_tracks) > 0:
        return sorted(poly_tracks, key=lambda track: len(track.notes))[-1]  # return track with most notes
    elif len(mono_tracks) > 0:
        pitch_means = [np.mean(x[1]) for x in mono_tracks]
        mono_tracks_ids = [x[0] for x in mono_tracks]
        if len(mono_tracks) >= 3 and max(pitch_means) - min(pitch_means) >= 8:
            return keep_long_notes(mf, new_program_id, mono_tracks_ids, 'Strings')
    else:
        return None