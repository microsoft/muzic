from tqdm import tqdm
import sqlite3
import argparse
import numpy as np
from magenta_chord_recognition import infer_chords_for_sequence, _key_chord_distribution, \
    _key_chord_transition_distribution, _CHORDS, _PITCH_CLASS_NAMES, NO_CHORD
import random
import os


class Item(object):
    def __init__(self, name, start, end, vel=0, pitch=0, track=0, value=''):
        self.name = name
        self.start = start  # start step
        self.end = end  # end step
        self.vel = vel
        self.pitch = pitch
        self.track = track
        self.value = value

    def __repr__(self):
        return f'Item(name={self.name:>10s}, start={self.start:>4d}, end={self.end:>4d}, ' \
               f'vel={self.vel:>3d}, pitch={self.pitch:>3d}, track={self.track:>2d}, ' \
               f'value={self.value:>10s})\n'

    def __eq__(self, other):
        return self.name == other.name and self.start == other.start and \
               self.pitch == other.pitch and self.track == other.track


def not_duplicate(bar1, bar2):
    notes1 = bar1.split(' ')[:-1][3::5]
    notes2 = bar2.split(' ')[:-1][3::5]
    return notes1 != notes2


def not_mono(bar):
    notes = bar.split(' ')[:-1][3::5]
    notes = [int(x[6:]) for x in notes]

    tmp = [0] * 128
    for idx in range(len(notes)):
        tmp[int(notes[idx])] = 1
    if (1 < len(notes) <= 3 and sum(tmp) == 1) or (len(notes) >= 4 and sum(tmp) < 3):
        return False
    return True


def get_chords(bar):
    notes = bar.split(' ')
    max_pos = 0
    note_items = []
    for i in range(len(notes) // 5):
        cadence = notes[5 * i]
        if notes[5 * i + 1][4:] == 'X':
            bar_idx = 0
        elif notes[5 * i + 1][4:] == 'Y':
            bar_idx = 1
        pos = int(notes[5 * i + 2][4:])
        pitch = int(notes[5 * i + 3][6:])
        dur = int(notes[5 * i + 4][4:])
        max_pos = max(max_pos, 16 * bar_idx + pos + dur)

        note_items.append(Item(
            name='On',
            start=16 * bar_idx + pos,
            end=16 * bar_idx + pos + dur,
            vel=100,
            pitch=pitch,
            track=0))
    note_items.sort(key=lambda x: (x.start, -x.end))
    pos_per_chord = 8
    max_chords = max(round(max_pos // pos_per_chord + 0.5), 1)
    chords = infer_chords_for_sequence(note_items,
                                       pos_per_chord=pos_per_chord,
                                       max_chords=max_chords,
                                       key_chord_loglik=key_chord_loglik,
                                       key_chord_transition_loglik=key_chord_transition_loglik
                                       )
    return ' '.join(chords)


def init():
    global key_chord_loglik, key_chord_transition_loglik
    chord_pitch_out_of_key_prob = 0.01
    key_change_prob = 0.001
    chord_change_prob = 0.5
    key_chord_distribution = _key_chord_distribution(
        chord_pitch_out_of_key_prob=chord_pitch_out_of_key_prob)
    key_chord_loglik = np.log(key_chord_distribution)
    key_chord_transition_distribution = _key_chord_transition_distribution(
        key_chord_distribution,
        key_change_prob=key_change_prob,
        chord_change_prob=chord_change_prob)
    key_chord_transition_loglik = np.log(key_chord_transition_distribution)


if __name__ == '__main__':
    init()
    parser = argparse.ArgumentParser(description='none.')
    parser.add_argument('notes_path')
    config = parser.parse_args()
    notes_path = config.notes_path
    # assert 'wc' in notes_path or 'wv' in notes_path, 'wrong file'

    if 'maj' in notes_path:
        MAJOR = 1
    else:
        MAJOR = 0

    if 'wc' in notes_path:
        CHORUS = 1
    else:
        CHORUS = 0

    if not os.path.exists('../database/ROC.db'):
        conn = sqlite3.connect('../database/ROC.db')
        print("Database connected.")
        c = conn.cursor()
        c.execute('''CREATE TABLE MELOLIB
               (LENGTH        INT    NOT NULL,
               CHORDS         TEXT     NOT NULL,
               MAJOR          INT     NOT NULL,
               CHORUS         INT     NOT NULL,
               NOTES          TEXT    NOT NULL);''')
        print("Table created.")
        conn.commit()
    else:
        conn = sqlite3.connect('../database/ROC.db')
        c = conn.cursor()
        print("database connected")

    with open(notes_path) as notes_file:
        melodies = notes_file.readlines()

        for melody in tqdm(melodies):
            notes = melody.split(' ')
            tmp1 = ''
            tmp2 = ''
            start_bar = int(notes[1][4:])
            for i in range(len(notes) // 5):
                cadence = notes[5 * i]
                bar_idx = int(notes[5 * i + 1][4:])
                pos = notes[5 * i + 2][4:]  # // pos_resolution
                pitch = int(notes[5 * i + 3][6:])
                dur = notes[5 * i + 4][4:]
                assert bar_idx >= start_bar
                if bar_idx - start_bar == 1 and not_mono(tmp1):
                    length = len(tmp1.split(' ')) // 5
                    chords = get_chords(tmp1)
                    assert len(chords) > 0, tmp1
                    c.execute(
                        "INSERT OR IGNORE INTO MELOLIB (LENGTH,MAJOR,CHORUS,NOTES,CHORDS) VALUES ('{}', '{}', '{}', '{}', '{}')".format(
                            length, MAJOR, CHORUS, tmp1, chords))
                if bar_idx == start_bar:
                    tmp1 += '{} bar_X Pos_{} Pitch_{} Dur_{} '.format(cadence, pos, pitch, dur)

                elif bar_idx == start_bar + 1:
                    tmp2 += '{} bar_Y Pos_{} Pitch_{} Dur_{} '.format(cadence, pos, pitch, dur)

            if tmp2 != '' and not_duplicate(tmp1, tmp2):
                # print(tmp1)
                # print(tmp2)
                tmp = tmp1 + tmp2
                if not_mono(tmp):
                    length = len(tmp.split(' ')) // 5
                    # input(tmp)
                    chords = get_chords(tmp)
                    assert len(chords) > 0, tmp
                    c.execute(
                        "INSERT OR IGNORE INTO MELOLIB (LENGTH,MAJOR,CHORUS,NOTES,CHORDS) VALUES ('{}', '{}', '{}', '{}', '{}')".format(
                            length, MAJOR, CHORUS, tmp, chords))
                    tmp1 = ''
                    tmp2 = ''
                    start_bar = bar_idx

            conn.commit()
        conn.close()
        # print(RHYTHM_SET)

