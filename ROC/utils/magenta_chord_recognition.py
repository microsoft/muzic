# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chord inference for NoteSequences."""
import bisect
import itertools
import math
import numbers

from absl import logging
import numpy as np

# Names of pitch classes to use (mostly ignoring spelling).
_PITCH_CLASS_NAMES = [
    'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Pitch classes in a key (rooted at zero).
_KEY_PITCHES = [0, 2, 4, 5, 7, 9, 11]

# Pitch classes in each chord kind (rooted at zero).
_CHORD_KIND_PITCHES = {
    '': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}
_CHORD_KINDS = _CHORD_KIND_PITCHES.keys()
NO_CHORD = 'N.C.'
# All usable chords, including no-chord.
_CHORDS = [NO_CHORD] + list(
    itertools.product(range(12), _CHORD_KINDS))

# All key-chord pairs.
_KEY_CHORDS = list(itertools.product(range(12), _CHORDS))

# Maximum length of chord sequence to infer.
_MAX_NUM_CHORDS = 1000

# MIDI programs that typically sound unpitched.
UNPITCHED_PROGRAMS = (
    list(range(96, 104)) + list(range(112, 120)) + list(range(120, 128)))

# Mapping from time signature to number of chords to infer per bar.
_DEFAULT_TIME_SIGNATURE_CHORDS_PER_BAR = {
    (2, 2): 1,
    (2, 4): 1,
    (3, 4): 1,
    (4, 4): 2,
    (6, 8): 2,
}


def _key_chord_distribution(chord_pitch_out_of_key_prob):
    """Probability distribution over chords for each key."""
    num_pitches_in_key = np.zeros([12, len(_CHORDS)], dtype=np.int32)
    num_pitches_out_of_key = np.zeros([12, len(_CHORDS)], dtype=np.int32)

    # For each key and chord, compute the number of chord notes in the key and the
    # number of chord notes outside the key.
    for key in range(12):
        key_pitches = set((key + offset) % 12 for offset in _KEY_PITCHES)
        for i, chord in enumerate(_CHORDS[1:]):
            root, kind = chord
            chord_pitches = set((root + offset) % 12
                                for offset in _CHORD_KIND_PITCHES[kind])
            num_pitches_in_key[key, i + 1] = len(chord_pitches & key_pitches)
            num_pitches_out_of_key[key, i +
                                   1] = len(chord_pitches - key_pitches)

    # Compute the probability of each chord under each key, normalizing to sum to
    # one for each key.
    mat = ((1 - chord_pitch_out_of_key_prob) ** num_pitches_in_key *
           chord_pitch_out_of_key_prob ** num_pitches_out_of_key)
    mat /= mat.sum(axis=1)[:, np.newaxis]
    return mat


def _key_chord_transition_distribution(
        key_chord_distribution, key_change_prob, chord_change_prob):
    """Transition distribution between key-chord pairs."""
    mat = np.zeros([len(_KEY_CHORDS), len(_KEY_CHORDS)])

    for i, key_chord_1 in enumerate(_KEY_CHORDS):
        key_1, chord_1 = key_chord_1
        chord_index_1 = i % len(_CHORDS)

        for j, key_chord_2 in enumerate(_KEY_CHORDS):
            key_2, chord_2 = key_chord_2
            chord_index_2 = j % len(_CHORDS)

            if key_1 != key_2:
                # Key change. Chord probability depends only on key and not previous
                # chord.
                mat[i, j] = (key_change_prob / 11)
                mat[i, j] *= key_chord_distribution[key_2, chord_index_2]

            else:
                # No key change.
                mat[i, j] = 1 - key_change_prob
                if chord_1 != chord_2:
                    # Chord probability depends on key, but we have to redistribute the
                    # probability mass on the previous chord since we know the chord
                    # changed.
                    mat[i, j] *= (
                        chord_change_prob * (
                            key_chord_distribution[key_2, chord_index_2] +
                            key_chord_distribution[key_2, chord_index_1] / (len(_CHORDS) -
                                                                            1)))
                else:
                    # No chord change.
                    mat[i, j] *= 1 - chord_change_prob

    return mat


def _chord_pitch_vectors():
    """Unit vectors over pitch classes for all chords."""
    x = np.zeros([len(_CHORDS), 12])
    for i, chord in enumerate(_CHORDS[1:]):
        root, kind = chord
        for offset in _CHORD_KIND_PITCHES[kind]:
            x[i + 1, (root + offset) % 12] = 1
    x[1:, :] /= np.linalg.norm(x[1:, :], axis=1)[:, np.newaxis]
    return x


def sequence_note_pitch_vectors(sequence, seconds_per_frame):
    """Compute pitch class vectors for temporal frames across a sequence.
    Args:
      sequence: The NoteSequence for which to compute pitch class vectors.
      seconds_per_frame: The size of the frame corresponding to each pitch class
          vector, in seconds. Alternatively, a list of frame boundary times in
          seconds (not including initial start time and final end time).
    Returns:
      A numpy array with shape `[num_frames, 12]` where each row is a unit-
      normalized pitch class vector for the corresponding frame in `sequence`.
    """
    frame_boundaries = sorted(seconds_per_frame)
    num_frames = len(frame_boundaries) + 1
    x = np.zeros([num_frames, 12])

    for note in sequence:
        # if note.is_drum:
        #     continue
        # if note.program in UNPITCHED_PROGRAMS:
        #     continue

        start_frame = bisect.bisect_right(frame_boundaries, note.start)
        end_frame = bisect.bisect_left(frame_boundaries, note.end)
        pitch_class = note.pitch % 12

        if start_frame >= end_frame:
            x[start_frame, pitch_class] += note.end - note.start
        else:
            x[start_frame, pitch_class] += (
                frame_boundaries[start_frame] - note.start)
            for frame in range(start_frame + 1, end_frame):
                x[frame, pitch_class] += (
                    frame_boundaries[frame] - frame_boundaries[frame - 1])
            x[end_frame, pitch_class] += (
                note.end - frame_boundaries[end_frame - 1])

    x_norm = np.linalg.norm(x, axis=1)
    nonzero_frames = x_norm > 0
    x[nonzero_frames, :] /= x_norm[nonzero_frames, np.newaxis]

    return x


def _chord_frame_log_likelihood(note_pitch_vectors, chord_note_concentration):
    """Log-likelihood of observing each frame of note pitches under each chord."""
    return chord_note_concentration * np.dot(note_pitch_vectors,
                                             _chord_pitch_vectors().T)


def _key_chord_viterbi(chord_frame_loglik,
                       key_chord_loglik,
                       key_chord_transition_loglik):
    """Use the Viterbi algorithm to infer a sequence of key-chord pairs."""
    num_frames, num_chords = chord_frame_loglik.shape
    num_key_chords = len(key_chord_transition_loglik)

    loglik_matrix = np.zeros([num_frames, num_key_chords])
    path_matrix = np.zeros([num_frames, num_key_chords], dtype=np.int32)

    # Initialize with a uniform distribution over keys.
    for i, key_chord in enumerate(_KEY_CHORDS):
        key, unused_chord = key_chord
        chord_index = i % len(_CHORDS)
        loglik_matrix[0, i] = (
            -np.log(12) + key_chord_loglik[key, chord_index] +
            chord_frame_loglik[0, chord_index])

    for frame in range(1, num_frames):
        # At each frame, store the log-likelihood of the best sequence ending in
        # each key-chord pair, along with the index of the parent key-chord pair
        # from the previous frame.
        mat = (np.tile(loglik_matrix[frame - 1][:, np.newaxis],
                       [1, num_key_chords]) +
               key_chord_transition_loglik)
        path_matrix[frame, :] = mat.argmax(axis=0)
        loglik_matrix[frame, :] = (
            mat[path_matrix[frame, :], range(num_key_chords)] +
            np.tile(chord_frame_loglik[frame], 12))

    # Reconstruct the most likely sequence of key-chord pairs.
    path = [np.argmax(loglik_matrix[-1])]
    for frame in range(num_frames, 1, -1):
        path.append(path_matrix[frame - 1, path[-1]])

    return [(index // num_chords, _CHORDS[index % num_chords])
            for index in path[::-1]]


class ChordInferenceError(Exception):  # pylint:disable=g-bad-exception-name
    pass


class SequenceAlreadyHasChordsError(ChordInferenceError):
    pass


class UncommonTimeSignatureError(ChordInferenceError):
    pass


class NonIntegerStepsPerChordError(ChordInferenceError):
    pass


class EmptySequenceError(ChordInferenceError):
    pass


class SequenceTooLongError(ChordInferenceError):
    pass


def infer_chords_for_sequence(sequence,
                              pos_per_chord,
                              max_chords,
                              key_chord_loglik=None,
                              key_chord_transition_loglik=None,
                              key_change_prob=0.001,
                              chord_change_prob=0.5,
                              chord_pitch_out_of_key_prob=0.01,
                              chord_note_concentration=100.0,
                              add_key_signatures=False):
    """Infer chords for a NoteSequence using the Viterbi algorithm.
      This uses some heuristics to infer chords for a quantized NoteSequence. At
      each chord position a key and chord will be inferred, and the chords will be
      added (as text annotations) to the sequence.
      If the sequence is quantized relative to meter, a fixed number of chords per
      bar will be inferred. Otherwise, the sequence is expected to have beat
      annotations and one chord will be inferred per beat.
      Args:
        sequence: The NoteSequence for which to infer chords. This NoteSequence will
            be modified in place.

        key_change_prob: Probability of a key change between two adjacent frames.
        chord_change_prob: Probability of a chord change between two adjacent
            frames.
        chord_pitch_out_of_key_prob: Probability of a pitch in a chord not belonging
            to the current key.
        chord_note_concentration: Concentration parameter for the distribution of
            observed pitches played over a chord. At zero, all pitches are equally
            likely. As concentration increases, observed pitches must match the
            chord pitches more closely.
        add_key_signatures: If True, also add inferred key signatures to
            `quantized_sequence` (and remove any existing key signatures).
      Raises:
        SequenceAlreadyHasChordsError: If `sequence` already has chords.
        QuantizationStatusError: If `sequence` is not quantized relative to
            meter but `chords_per_bar` is specified or no beat annotations are
            present.
        UncommonTimeSignatureError: If `chords_per_bar` is not specified and
            `sequence` is quantized and has an uncommon time signature.
        NonIntegerStepsPerChordError: If the number of quantized steps per chord
            is not an integer.
        EmptySequenceError: If `sequence` is empty.
        SequenceTooLongError: If the number of chords to be inferred is too
            large.
    """
    beats = [pos_per_chord * i for i in range(max_chords)]
    if len(beats) == 0:
        raise Exception('max chords should > 0')
    num_chords = len(beats)
    if num_chords > _MAX_NUM_CHORDS:
        raise Exception(
            'NoteSequence too long for chord inference: %d frames' % num_chords)

    # Compute pitch vectors for each chord frame, then compute log-likelihood of
    # observing those pitch vectors under each possible chord.
    note_pitch_vectors = sequence_note_pitch_vectors(
        sequence,
        beats)
    chord_frame_loglik = _chord_frame_log_likelihood(
        note_pitch_vectors, chord_note_concentration)

    # Compute distribution over chords for each key, and transition distribution
    # between key-chord pairs.
    if key_chord_loglik is None:
        key_chord_distribution = _key_chord_distribution(
            chord_pitch_out_of_key_prob=chord_pitch_out_of_key_prob)
        key_chord_loglik = np.log(key_chord_distribution)

    if key_chord_transition_loglik is None:
        key_chord_transition_distribution = _key_chord_transition_distribution(
            key_chord_distribution,
            key_change_prob=key_change_prob,
            chord_change_prob=chord_change_prob)
        key_chord_transition_loglik = np.log(key_chord_transition_distribution)

    key_chords = _key_chord_viterbi(
        chord_frame_loglik, key_chord_loglik, key_chord_transition_loglik)

    # if add_key_signatures:
    #     del sequence.key_signatures[:]

    # Add the inferred chord changes to the sequence, optionally adding key
    # signature(s) as well.
    # current_key_name = None
    # current_chord_name = None
    chords = []
    for frame, (key, chord) in enumerate(key_chords):
        # time = beats[frame]

        # if _PITCH_CLASS_NAMES[key] != current_key_name:
        #     # A key change was inferred.
        #     if add_key_signatures:
        #         ks = sequence.key_signatures.add()
        #         ks.time = time
        #         ks.key = key
        #     else:
        #         if current_key_name is not None:
        #             logging.info(
        #                 'Sequence has key change from %s to %s at %f seconds.',
        #                 current_key_name, _PITCH_CLASS_NAMES[key], time)
        #
        # current_key_name = _PITCH_CLASS_NAMES[key]
        if chord == NO_CHORD:
            figure = NO_CHORD
        else:
            root, kind = chord
            figure = '%s:%s' % (_PITCH_CLASS_NAMES[root], kind)
        chords.append(figure)
    return chords
    # if figure != current_chord_name:
    #     ta = sequence.text_annotations.add()
    #     ta.time = time
    #     ta.quantized_step = 0 if frame == 0 else sorted_beat_steps[frame - 1]
    #     ta.text = figure
    # current_chord_name = figure
