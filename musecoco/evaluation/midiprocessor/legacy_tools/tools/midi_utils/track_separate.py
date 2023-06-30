# from __future__ import print_function

import pretty_midi
import pypianoroll
import music21

import numpy as np
import pickle
import pandas as pd
import scipy.stats
from collections import Counter
from functools import reduce
import os
from copy import deepcopy
import sys
import argparse
import json
import pdb


def remove_empty_track(midi_file):
    '''
    1. read pretty midi data
    2. remove emtpy track,
    also remove track with fewer than 10% notes of the track
    with most notes

    ********
    Return: pretty_midi object, pypianoroll object
    '''
    # pdb.set_trace()
    try:
        pretty_midi_data = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f'exceptions in read the file {midi_file}')
        return None, None

    #     print('I00:', pretty_midi_data.instruments)
    pypiano_data = pypianoroll.Multitrack()

    try:
        pypiano_data.parse_pretty_midi(
            pretty_midi_data, skip_empty_tracks=False)
    except Exception as e:
        print(f'exceptions for pypianoroll in read the file {midi_file}')
        return None, None

    drum_idx = []
    for i, instrument in enumerate(pretty_midi_data.instruments):
        if instrument.is_drum:
            drum_idx.append(i)

    note_count = [np.count_nonzero(np.any(track.pianoroll, axis=1)) \
                  for track in pypiano_data.tracks]

    empty_indices = np.array(note_count) < 10
    remove_indices = np.arange(len(pypiano_data.tracks))[empty_indices]

    for index in sorted(remove_indices, reverse=True):
        del pypiano_data.tracks[index]
        del pretty_midi_data.instruments[index]
    return pretty_midi_data, pypiano_data


def remove_duplicate_tracks(features, replace=False):
    if not replace:
        features = features.copy()

    file_names = features.file_names.unique()
    duplicates = []

    for file_name in file_names:
        file_features = features[features.file_names == file_name]
        number_notes = Counter(file_features.num_notes)
        notes = []
        for ele in number_notes:
            if number_notes[ele] > 1:
                notes.append(ele)
        h_pits = []
        for note in notes:
            number_h_pit = Counter(file_features[file_features.num_notes == note].h_pit)

            for ele in number_h_pit:
                if number_h_pit[ele] > 1:
                    h_pits.append(ele)

        l_pits = []
        for h_pit in h_pits:
            number_l_pit = Counter(file_features[file_features.h_pit == h_pit].l_pit)

            for ele in number_l_pit:
                if number_l_pit[ele] > 1:
                    l_pits.append(ele)

        notes = list(set(notes))
        h_pits = list(set(h_pits))
        l_pits = list(set(l_pits))

        for note in notes:
            note_index = file_features[file_features.num_notes == note].index.values
            for h_pit in h_pits:
                h_pit_index = file_features[file_features.h_pit == h_pit].index.values
                for l_pit in l_pits:
                    l_pit_index = file_features[file_features.l_pit == l_pit].index.values

                    index_intersect = reduce(np.intersect1d, (note_index, h_pit_index, l_pit_index))

                    if len(index_intersect) > 1:
                        duplicates.append(index_intersect)

    ### copy the labels in the tracks to be removed

    melody_track_name = ['sing', 'vocals', 'vocal', 'melody', 'melody:']
    bass_track_name = ['bass', 'bass:']
    chord_track_name = ['chord', 'chords', 'harmony']

    for indices in duplicates:
        melody_track = False
        bass_track = False
        chord_track = False
        labels = features.loc[indices, 'trk_names']
        for label in labels:
            if label in melody_track_name:
                melody_track = True

            elif label in bass_track_name:

                bass_track = True

            elif label in chord_track_name:
                chord_track = True
            else:
                pass

        if melody_track:
            features.loc[indices, 'trk_names'] = 'melody'
        if bass_track:
            features.loc[indices, 'trk_names'] = 'bass'
        if chord_track:
            features.loc[indices, 'trk_names'] = 'chord'

        features.drop(indices[1:], inplace=True)
        print(indices[1:])

    return features


def remove_file_duplicate_tracks(features, pm):
    duplicates = []

    index_to_remove = []
    number_notes = Counter(features.num_notes)
    notes = []
    for ele in number_notes:  # 统计是否有 notes 数量相等的 track
        if number_notes[ele] > 1:
            notes.append(ele)
    h_pits = []
    for note in notes:
        number_h_pit = Counter(features[features.num_notes == note].h_pit)

        for ele in number_h_pit:
            if number_h_pit[ele] > 1:
                h_pits.append(ele)

    l_pits = []
    for h_pit in h_pits:
        number_l_pit = Counter(features[features.h_pit == h_pit].l_pit)

        for ele in number_l_pit:
            if number_l_pit[ele] > 1:
                l_pits.append(ele)

    notes = list(set(notes))
    h_pits = list(set(h_pits))
    l_pits = list(set(l_pits))

    for note in notes:
        note_index = features[features.num_notes == note].index.values
        for h_pit in h_pits:
            h_pit_index = features[features.h_pit == h_pit].index.values
            for l_pit in l_pits:
                l_pit_index = features[features.l_pit == l_pit].index.values

                index_intersect = reduce(np.intersect1d, (note_index, h_pit_index, l_pit_index))

                if len(index_intersect) > 1:
                    duplicates.append(index_intersect)

    ### copy the labels in the tracks to be removed

    melody_track_name = ['sing', 'vocals', 'vocal', 'melody', 'melody:']
    bass_track_name = ['bass', 'bass:']
    chord_track_name = ['chord', 'chords', 'harmony']

    for indices in duplicates:
        features.drop(indices[1:], inplace=True)
        for index in indices[1:]:
            index_to_remove.append(index)

    indices = np.sort(np.array(index_to_remove))

    for index in indices[::-1]:
        del pm.instruments[index]

    features.reset_index(inplace=True, drop='index')

    return


def walk(folder_name):
    files = []
    for p, d, f in os.walk(folder_name):
        for file in f:
            base_name = os.path.basename(file)
            base_name = os.path.splitext(base_name)[0]
            if base_name.isupper():
                continue
            if file.endswith('.mid') or file.endswith('.MID'):
                files.append(os.path.join(p, file))
    return files


def relative_duration(pianoroll_data):
    '''

    read pianoroll_data data

    '''

    note_count = [np.count_nonzero(np.any(track.pianoroll, axis=1)) \
                  for track in pianoroll_data.tracks]  # 每个轨道持续时间

    relative_durations = note_count / np.max(note_count)

    relative_durations = np.array(relative_durations)

    relative_durations = relative_durations[:, np.newaxis]

    assert relative_durations.shape == (len(pianoroll_data.tracks), 1)

    return np.array(relative_durations)


def number_of_notes(pretty_midi_data):
    '''
    read pretty-midi data
    '''
    number_of_notes = []
    for instrument in pretty_midi_data.instruments:
        number_of_notes.append(len(instrument.notes))

    number_of_notes = np.array(number_of_notes, dtype='uint16')

    number_of_notes = number_of_notes[:, np.newaxis]

    assert number_of_notes.shape == (len(pretty_midi_data.instruments), 1)

    return number_of_notes


def occupation_rate(pianoroll_data):
    '''
    read pypianoroll data
    '''

    occup_rates = []
    for track in pianoroll_data.tracks:
        piano_roll = track.pianoroll
        occup_rates.append(np.count_nonzero(np.any(piano_roll, 1)) / piano_roll.shape[0])

    occup_rates = np.array(occup_rates)
    occup_rates = occup_rates[:, np.newaxis]

    assert occup_rates.shape == (len(pianoroll_data.tracks), 1)
    return occup_rates


def polyphony_rate(pianoroll_data):
    '''
    use pianoroll data

    '''

    rates = []
    for track in pianoroll_data.tracks:
        piano_roll = track.pianoroll
        number_poly_note = np.count_nonzero(np.count_nonzero(piano_roll, 1) > 1)

        rate = number_poly_note / np.count_nonzero(np.any(piano_roll, 1))
        rates.append(rate)

    rates = np.array(rates)
    rates = rates[:, np.newaxis]

    assert rates.shape == (len(pianoroll_data.tracks), 1)
    return rates


def pitch(pianoroll_data):
    '''
    read pypianoroll data

    Returns
        -------
        a numpy array in the shape of (number of tracks, 8)

        the 8 columns are highest pitch, lowest pitch, pitch mode, pitch std,
        and the norm value across different tracks for those values

    '''

    highest = []
    lowest = []
    modes = []
    stds = []

    # pitches = np.array([note.pitch] for note in instrument.notes)

    def array_creation_by_count(counts):
        result = []
        for i, count in enumerate(counts):
            if count != 0:
                result.append([i] * count)

        result = np.array([item for sublist in result for item in sublist])
        return result

    for track in pianoroll_data.tracks:
        highest_note = np.where(np.any(track.pianoroll, 0))[0][-1]
        lowest_note = np.where(np.any(track.pianoroll, 0))[0][0]
        pitch_array = array_creation_by_count(np.count_nonzero(track.pianoroll, 0))

        mode_pitch = scipy.stats.mode(pitch_array)  # 返回出现最多的音高
        mode_pitch = mode_pitch.mode[0]

        # print(mode_pitch)

        std_pitch = np.std(pitch_array)

        # print(std_pitch)

        highest.append(highest_note)
        lowest.append(lowest_note)
        modes.append(mode_pitch)
        stds.append(std_pitch)

    highest = np.array(highest, dtype='uint8')
    lowest = np.array(lowest, dtype='uint8')
    modes = np.array(modes, dtype='uint8')
    stds = np.array(stds, dtype='float32')

    if np.max(highest) - np.min(highest) == 0:
        highest_norm = np.ones_like(highest)
    else:

        highest_norm = (highest - np.min(highest)) / (np.max(highest) - np.min(highest))

    if np.max(lowest) - np.min(lowest) == 0:
        lowest_norm = np.zeros_like(lowest)
    else:
        lowest_norm = (lowest - np.min(lowest)) / (np.max(lowest) - np.min(lowest))

    if np.max(modes) - np.min(modes) == 0:
        modes_norm = np.zeros_like(modes)
    else:
        modes_norm = (modes - np.min(modes)) / (np.max(modes) - np.min(modes))

    if np.max(stds) - np.min(stds) == 0:
        stds_norm = np.zeros_like(stds)
    else:
        stds_norm = (stds - np.min(stds)) / (np.max(stds) - np.min(stds))

    result = np.vstack((highest, lowest, modes, stds, highest_norm, lowest_norm, modes_norm, stds_norm))
    result = result.T

    # print(result.shape)
    assert result.shape == (len(pianoroll_data.tracks), 8)

    return result


def pitch_intervals(pretty_midi_data):
    '''
    use pretty-midi data here

     Returns
        -------
        a numpy array in the shape of (number of tracks, 5)

        the 5 columns are number of different intervals, largest interval,
        smallest interval, mode interval and interval std of this track,
        and the norm value across different tracks for those values

    '''

    different_interval = []
    largest_interval = []
    smallest_interval = []
    mode_interval = []
    std_interval = []

    def get_intervals(notes, threshold=-1):
        '''

        threshold is the second for the space between two consecutive notes
        '''

        intervals = []
        for i in range(len(notes) - 1):
            note1 = notes[i]
            note2 = notes[i + 1]

            if note1.end - note2.start >= threshold:
                if note2.end >= note1.end:
                    intervals.append(abs(note2.pitch - note1.pitch))
        return np.array(intervals)

    for instrument in pretty_midi_data.instruments:
        intervals = get_intervals(instrument.notes, -3)
        #         print(f'intervals is {intervals}')

        if len(intervals) > 0:

            different_interval.append(len(np.unique(intervals)))
            largest_interval.append(np.max(intervals))
            smallest_interval.append(np.min(intervals))
            mode_interval.append(scipy.stats.mode(intervals).mode[0])
            std_interval.append(np.std(intervals))
        else:
            different_interval.append(-1)
            largest_interval.append(-1)
            smallest_interval.append(-1)
            mode_interval.append(-1)
            std_interval.append(-1)

    different_interval = np.array(different_interval, dtype='uint8')
    largest_interval = np.array(largest_interval, dtype='uint8')
    smallest_interval = np.array(smallest_interval, dtype='uint8')
    mode_interval = np.array(mode_interval, dtype='uint8')
    std_interval = np.array(std_interval, dtype='float32')

    if np.max(different_interval) - np.min(different_interval) == 0:
        different_interval_norm = np.zeros_like(different_interval)
    else:
        different_interval_norm = (different_interval - np.min(different_interval)) / (
                np.max(different_interval) - np.min(different_interval))

    if np.max(largest_interval) - np.min(largest_interval) == 0:
        largest_interval_norm = np.ones_like(largest_interval)
    else:
        largest_interval_norm = (largest_interval - np.min(largest_interval)) / (
                np.max(largest_interval) - np.min(largest_interval))

    if np.max(smallest_interval) - np.min(smallest_interval) == 0:
        smallest_interval_norm = np.zeros_like(smallest_interval)
    else:
        smallest_interval_norm = (smallest_interval - np.min(smallest_interval)) / (
                np.max(smallest_interval) - np.min(smallest_interval))

    if np.max(mode_interval) - np.min(mode_interval) == 0:
        mode_interval_norm = np.zeros_like(mode_interval)
    else:
        mode_interval_norm = (mode_interval - np.min(mode_interval)) / (np.max(mode_interval) - np.min(mode_interval))

    if np.max(std_interval) - np.min(std_interval) == 0:
        std_interval_norm = np.zeros_like(std_interval)
    else:
        std_interval_norm = (std_interval - np.min(std_interval)) / (np.max(std_interval) - np.min(std_interval))

    result = np.vstack((different_interval, largest_interval, smallest_interval, \
                        mode_interval, std_interval, different_interval_norm, \
                        largest_interval_norm, smallest_interval_norm, \
                        mode_interval_norm, std_interval_norm))

    result = result.T

    assert (result.shape == (len(pretty_midi_data.instruments), 10))

    return result


def note_durations(pretty_midi_data):
    '''
    use pretty-midi data here

    Parameters
        ----------
        data : pretty-midi data

         Returns
        -------
        a numpy array in the shape of (number of tracks, 4)

        the 4 columns are longest, shortest, mean, std of note durations
        and the norm value across different tracks for those values
    '''

    longest_duration = []
    shortest_duration = []
    mean_duration = []
    std_duration = []

    for instrument in pretty_midi_data.instruments:
        notes = instrument.notes
        durations = np.array([note.end - note.start for note in notes])

        # print(f'durations is {durations}')

        longest_duration.append(np.max(durations))
        shortest_duration.append(np.min(durations))
        mean_duration.append(np.mean(durations))
        std_duration.append(np.std(durations))

    longest_duration = np.array(longest_duration)
    shortest_duration = np.array(shortest_duration)
    mean_duration = np.array(mean_duration)
    std_duration = np.array(std_duration)

    if np.max(longest_duration) - np.min(longest_duration) == 0:
        longest_duration_norm = np.ones_like(longest_duration)
    else:
        longest_duration_norm = (longest_duration - np.min(longest_duration)) / (
                np.max(longest_duration) - np.min(longest_duration))

    if np.max(shortest_duration) - np.min(shortest_duration) == 0:
        shortest_duration_norm = np.zeros_like(shortest_duration)
    else:
        shortest_duration_norm = (shortest_duration - np.min(shortest_duration)) / (
                np.max(shortest_duration) - np.min(shortest_duration))

    if np.max(mean_duration) - np.min(mean_duration) == 0:
        mean_duration_norm = np.zeros_like(mean_duration)
    else:
        mean_duration_norm = (mean_duration - np.min(mean_duration)) / (np.max(mean_duration) - np.min(mean_duration))

    if np.max(std_duration) - np.min(std_duration) == 0:
        std_duration_norm = np.zeros_like(std_duration)
    else:
        std_duration_norm = (std_duration - np.min(std_duration)) / (np.max(std_duration) - np.min(std_duration))

    result = np.vstack((longest_duration, shortest_duration, mean_duration, \
                        std_duration, longest_duration_norm, shortest_duration_norm, \
                        mean_duration_norm, std_duration_norm))

    result = result.T

    # print(result.shape)

    assert result.shape == (len(pretty_midi_data.instruments), 8)
    return result


def note_durations(pretty_midi_data):
    '''
    use pretty-midi data here

    Parameters
        ----------
        data : pretty-midi data

         Returns
        -------
        a numpy array in the shape of (number of tracks, 4)

        the 4 columns are longest, shortest, mean, std of note durations
        and the norm value across different tracks for those values
    '''

    longest_duration = []
    shortest_duration = []
    mean_duration = []
    std_duration = []

    for instrument in pretty_midi_data.instruments:
        notes = instrument.notes
        durations = np.array([note.end - note.start for note in notes])

        longest_duration.append(np.max(durations))
        shortest_duration.append(np.min(durations))
        mean_duration.append(np.mean(durations))
        std_duration.append(np.std(durations))

    longest_duration = np.array(longest_duration)
    shortest_duration = np.array(shortest_duration)
    mean_duration = np.array(mean_duration)
    std_duration = np.array(std_duration)

    if np.max(longest_duration) - np.min(longest_duration) == 0:
        longest_duration_norm = np.ones_like(longest_duration)
    else:
        longest_duration_norm = (longest_duration - np.min(longest_duration)) / (
                np.max(longest_duration) - np.min(longest_duration))

    if np.max(shortest_duration) - np.min(shortest_duration) == 0:
        shortest_duration_norm = np.zeros_like(shortest_duration)
    else:
        shortest_duration_norm = (shortest_duration - np.min(shortest_duration)) / (
                np.max(shortest_duration) - np.min(shortest_duration))

    if np.max(mean_duration) - np.min(mean_duration) == 0:
        mean_duration_norm = np.zeros_like(mean_duration)
    else:
        mean_duration_norm = (mean_duration - np.min(mean_duration)) / (np.max(mean_duration) - np.min(mean_duration))

    if np.max(std_duration) - np.min(std_duration) == 0:
        std_duration_norm = np.zeros_like(std_duration)
    else:
        std_duration_norm = (std_duration - np.min(std_duration)) / (np.max(std_duration) - np.min(std_duration))

    result = np.vstack((longest_duration, shortest_duration, mean_duration, \
                        std_duration, longest_duration_norm, shortest_duration_norm, \
                        mean_duration_norm, std_duration_norm))

    result = result.T

    # print(result.shape)

    assert result.shape == (len(pretty_midi_data.instruments), 8)
    return result


def all_features(midi_file):
    '''
    compute 34 features from midi data. Each track of each song have 30 features

    1 set of feature:
    duration, number of notes, occupation rate, polyphony rate,

    2 set of feature:
    Highest pitch, lowest pitch, pitch mode, pitch std,
    Highest pitch norm, lowest pitch norm, pitch mode norm, pitch std norm

    3 set of feature

    number of interval, largest interval,
    smallest interval, interval mode,
    number of interval norm, largest interval norm,
    smallest interval norm, interval mode norm

    4 set of feature

    longest note duration, shortest note duration,
     mean note duration, note duration std,
     longest note duration norm, shortest note duration norm,
     mean note duration norm, note duration std norm

    for all the normed feature,  it is the normalised features
    across different tracks within a midi file

    5 set of feature:
    track_programs,track_names,file_names,is_drum

    '''

    pm, pypiano = remove_empty_track(midi_file)

    if pm is None:
        return None

    if len(pypiano.tracks) != len(pm.instruments):
        # print(f'pypiano track length is {len(pypiano.tracks)} does not equal \
        #       to pretty_midi length {len(pm.instruments)} in file {midi_file}')
        return None

    # print(f'the file is {midi_file}')

    track_programs = np.array([i.program for i in pm.instruments])[:, np.newaxis]
    track_names = []

    try:
        for instrument in pm.instruments:
            if len(instrument.name) > 1:
                track_names.append(instrument.name.rsplit()[0].lower())
            #             if instrument.name.strip() is not '':
            #                 track_names.append(instrument.name.rsplit()[0].lower())
            else:
                track_names.append('')

    except Exception as e:
        print(f'exceptions in find instrument name {midi_file}')
        return None

    track_names = np.array(track_names)[:, np.newaxis]
    file_names = np.array([midi_file] * len(pm.instruments))[:, np.newaxis]
    is_drum = np.array([i.is_drum for i in pm.instruments])[:, np.newaxis]

    rel_durations = relative_duration(pypiano)
    number_notes = number_of_notes(pm)
    occup_rate = occupation_rate(pypiano)
    poly_rate = polyphony_rate(pypiano)

    pitch_features = pitch(pypiano)

    pitch_interval_features = pitch_intervals(pm)

    note_duration_features = note_durations(pm)

    all_features = np.hstack((track_programs, track_names, file_names, is_drum, \
                              rel_durations, number_notes, occup_rate, \
                              poly_rate, pitch_features, \
                              pitch_interval_features, note_duration_features
                              ))

    # print(all_features.shape)
    assert all_features.shape == (len(pm.instruments), 34)

    return all_features


def cal_file_features(midi_file):
    '''
    compute 34 features from midi data. Each track of each song have 30 features

    1 set of feature:
    duration, number of notes, occupation rate, polyphony rate,

    2 set of feature:
    Highest pitch, lowest pitch, pitch mode, pitch std,
    Highest pitch norm, lowest pitch norm, pitch mode norm, pitch std norm

    3 set of feature

    number of interval, largest interval,
    smallest interval, interval mode,
    number of interval norm, largest interval norm,
    smallest interval norm, interval mode norm

    4 set of feature

    longest note duration, shortest note duration,
     mean note duration, note duration std,
     longest note duration norm, shortest note duration norm,
     mean note duration norm, note duration std norm

    for all the normed feature,  it is the normalised features
    across different tracks within a midi file

    5 set of feature:
    track_programs,track_names,file_names,is_drum

    '''

    # pdb.set_trace()
    pm, pypiano = remove_empty_track(midi_file)

    if pm is None:
        return None, None

    if len(pypiano.tracks) != len(pm.instruments):
        print(f'pypiano track length is {len(pypiano.tracks)} does not equal \
              to pretty_midi length {len(pm.instruments)} in file {midi_file}')
        return None

    # print(f'the file is {midi_file}')

    '''
        5 set of feature
    '''
    track_programs = np.array([i.program for i in pm.instruments])[:, np.newaxis]
    track_names = []

    try:
        for instrument in pm.instruments:
            if len(instrument.name.rsplit()) > 0:
                track_names.append(instrument.name.rsplit()[0].lower())
            #             if instrument.name.strip() is not '':
            #                 track_names.append(instrument.name.rsplit()[0].lower())
            else:
                track_names.append('')

    except Exception as e:
        print(f'exceptions in find instrument name {midi_file}')
        return None

    #     basename = os.path.basename(midi_file)
    #     pm.write('/Users/ruiguo/Downloads/2000midi/new/' + basename)

    track_names = np.array(track_names)[:, np.newaxis]
    file_names = np.array([midi_file] * len(pm.instruments))[:, np.newaxis]
    is_drum = np.array([i.is_drum for i in pm.instruments])[:, np.newaxis]
    # print(is_drum)

    '''
        1 set of feature
    '''
    rel_durations = relative_duration(pypiano)  # 相对持续时间， 每个轨道相对占比
    number_notes = number_of_notes(pm)
    occup_rate = occupation_rate(pypiano)
    poly_rate = polyphony_rate(pypiano)

    '''
        2 set of feature
    '''
    pitch_features = pitch(pypiano)

    '''
        3 set of feature
    '''
    pitch_interval_features = pitch_intervals(pm)  # 音程变化 信息
    # pdb.set_trace()

    '''
        4 set of feature
    '''
    note_duration_features = note_durations(pm)

    all_features = np.hstack((track_programs, track_names, file_names, is_drum, \
                              rel_durations, number_notes, occup_rate, \
                              poly_rate, pitch_features, \
                              pitch_interval_features, note_duration_features
                              ))

    # print(all_features.shape)
    assert all_features.shape == (len(pm.instruments), 34)

    return all_features, pm


melody_track_name = ['sing', 'vocals', 'vocal', 'melody', 'melody:']
bass_track_name = ['bass', 'bass:']
chord_track_name = ['chord', 'chords', 'harmony']
check_melody = lambda x: x in melody_track_name
check_bass = lambda x: x in bass_track_name
check_chord = lambda x: x in chord_track_name

columns = ['trk_prog', 'trk_names', 'file_names', 'is_drum',
           'dur', 'num_notes', 'occup_rate', 'poly_rate',
           'h_pit', 'l_pit', 'pit_mode', 'pit_std',
           'h_pit_nor', 'l_pit_nor', 'pit_mode_nor', 'pit_std_nor',
           'num_intval', 'l_intval', 's_intval', 'intval_mode', 'intval_std',
           'num_intval_nor', 'l_intval_nor', 's_intval_nor', 'intval_mode_nor', 'intval_std_nor',
           'l_dur', 's_dur', 'mean_dur', 'dur_std',
           'l_dur_nor', 's_dur_nor', 'mean_dur_nor', 'dur_std_nor']

boolean_dict = {'True': True, 'False': False}


def add_labels(features):
    features = pd.DataFrame(features, columns=columns)

    for name in columns[4:]:
        features[name] = pd.to_numeric(features[name])

    features['trk_prog'] = pd.to_numeric(features['trk_prog'])
    features['is_drum'] = features['is_drum'].map(boolean_dict)

    return features


def predict_labels(features, melody_model, bass_model, chord_model):
    temp_features = features.copy()
    temp_features = temp_features.drop(temp_features.columns[:4], axis=1)

    predicted_melody = melody_model.predict(temp_features)
    predicted_bass = bass_model.predict(temp_features)
    predicted_chord = chord_model.predict(temp_features)

    features['is_melody'] = list(map(check_melody, features['trk_names']))
    features['is_bass'] = list(map(check_bass, features['trk_names']))
    features['is_chord'] = list(map(check_chord, features['trk_names']))

    features['melody_predict'] = predicted_melody
    features['bass_predict'] = predicted_bass
    features['chord_predict'] = predicted_chord
    return features
