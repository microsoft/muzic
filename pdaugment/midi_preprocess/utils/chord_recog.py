"""Chord inference for NoteSequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import itertools
import math
import numbers
import miditoolkit
import numpy as np
from utils.midi_io import Item

# Names of pitch classes to use (mostly ignoring spelling).
_PITCH_CLASS_NAMES = [
    'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Pitch classes in a key (rooted at zero).
_KEY_PITCHES = [0, 2, 4, 5, 7, 9, 11]

# MIDI programs that typically sound unpitched.
UNPITCHED_PROGRAMS = (
        list(range(96, 104)) + list(range(112, 120)) + list(range(120, 128)))

# Chord symbol for "no chord".
NO_CHORD = 'N.C.'

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

# All usable chords, including no-chord.
_CHORDS = [NO_CHORD] + list(itertools.product(range(12), _CHORD_KINDS))

# All key-chord pairs.
_KEY_CHORDS = list(itertools.product(range(12), _CHORDS))

# Maximum length of chord sequence to infer.
_MAX_NUM_CHORDS = 10000

# Mapping from time signature to number of chords to infer per bar.
_DEFAULT_TIME_SIGNATURE_CHORDS_PER_BAR = {
    (2, 2): 1,
    (2, 4): 1,
    (3, 4): 1,
    (4, 4): 4,
    # (4, 4): 2,
    (6, 8): 2,
}


def steps_per_bar_in_quantized_sequence(steps_per_quarter=4):
    """Calculates steps per bar in a NoteSequence that has been quantized. assume time signature is 4/4
    Returns:
        Steps per bar as a floating point number.
        """
    quarters_per_beat = 4.0 / 4
    quarters_per_bar = (quarters_per_beat * 4)
    steps_per_bar_float = (steps_per_quarter * quarters_per_bar)
    return steps_per_bar_float


def steps_per_quarter_to_steps_per_second(steps_per_quarter=4, qpm=120):
    """Calculates steps per second given steps_per_quarter and a qpm."""
    return steps_per_quarter * qpm / 60.0


def tick2second(tick, ticks_per_beat=480, tempo=120):
    """Convert absolute time in ticks to seconds.
    Returns absolute time in seconds for a chosen MIDI file time
    resolution (ticks per beat, also called PPQN or pulses per quarter
    note) and tempo (microseconds per beat).
         240 => 250000
        120 => 500000
        60 => 1000000
    """
    scale = int(round((60 * 1000000) / tempo)) * 1e-6 / ticks_per_beat
    return tick * scale


def second2tick(second, ticks_per_beat=480, tempo=120):
    """Convert absolute time in seconds to ticks.
    Returns absolute time in ticks for a chosen MIDI file time
    resolution (ticks per beat, also called PPQN or pulses per quarter
    note) and tempo (microseconds per beat).
    """
    scale = round((60 * 1000000) / tempo) * 1e-6 / ticks_per_beat
    return int(second / scale)


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
            num_pitches_out_of_key[key, i + 1] = len(chord_pitches - key_pitches)

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


def sequence_note_pitch_vectors(sequence, seconds_per_frame, tempo=120):
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
    if isinstance(seconds_per_frame, numbers.Number):
        # Construct array of frame boundary times.
        num_frames = int(math.ceil(tick2second(sequence[-1].end, tempo=tempo) / seconds_per_frame))
        # print(f"num_frames:{num_frames}")
        frame_boundaries = seconds_per_frame * np.arange(1, num_frames)
    else:
        frame_boundaries = sorted(seconds_per_frame)
        num_frames = len(frame_boundaries) + 1

    x = np.zeros([num_frames, 12])

    for note in sequence:
        start_frame = bisect.bisect_right(frame_boundaries, tick2second(note.start, tempo=tempo))
        end_frame = bisect.bisect_left(frame_boundaries, tick2second(note.end, tempo=tempo))
        pitch_class = note.pitch % 12

        if start_frame >= end_frame:
            x[start_frame, pitch_class] += tick2second(note.end, tempo=tempo) - tick2second(note.start, tempo=tempo)
        else:
            x[start_frame, pitch_class] += (
                    frame_boundaries[start_frame] - tick2second(note.start, tempo=tempo))
            for frame in range(start_frame + 1, end_frame):
                x[frame, pitch_class] += (
                        frame_boundaries[frame] - frame_boundaries[frame - 1])
            x[end_frame, pitch_class] += (
                    tick2second(note.end, tempo=tempo) - frame_boundaries[end_frame - 1])

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
                              tempo=120,
                              ticks_per_beat=None,
                              chords_per_bar=None,
                              key_change_prob=0.001,
                              chord_change_prob=0.5,
                              chord_pitch_out_of_key_prob=0.01,
                              chord_note_concentration=100.0):
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
      chords_per_bar: If `sequence` is quantized, the number of chords per bar to
          infer. If None, use a default number of chords based on the time
          signature of `sequence`.
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
    # Infer a fixed number of chords per bar.
    tempo = tempo
    ticks_per_beat = ticks_per_beat
    if chords_per_bar is None:
        chords_per_bar = _DEFAULT_TIME_SIGNATURE_CHORDS_PER_BAR[(4, 4)]

    # Determine the number of seconds (and steps) each chord is held.
    # steps_per_bar_float = steps_per_bar_in_quantized_sequence(steps_per_quarter=steps_per_quarter)
    # steps_per_chord_float = steps_per_bar_float / chords_per_bar
    # if steps_per_chord_float != round(steps_per_chord_float):
    #     raise NonIntegerStepsPerChordError(
    #         'Non-integer number of steps per chord: %f' % steps_per_chord_float)
    seconds_per_chord = tick2second(ticks_per_beat * 4 // chords_per_bar, tempo=tempo)
    num_chords = int(math.ceil(tick2second(sequence[-1].end, tempo=tempo) / seconds_per_chord))
    if num_chords == 0:
        raise EmptySequenceError('empty midi')

    if num_chords > _MAX_NUM_CHORDS:
        raise SequenceTooLongError(
            'too long for chord inference: %d frames' % num_chords)

    # Compute pitch vectors for each chord frame, then compute log-likelihood of
    # observing those pitch vectors under each possible chord.
    note_pitch_vectors = sequence_note_pitch_vectors(sequence, seconds_per_chord, tempo=tempo)
    chord_frame_loglik = _chord_frame_log_likelihood(note_pitch_vectors, chord_note_concentration)

    # Compute distribution over chords for each key, and transition distribution
    # between key-chord pairs.
    key_chord_distribution = _key_chord_distribution(
        chord_pitch_out_of_key_prob=chord_pitch_out_of_key_prob)
    key_chord_transition_distribution = _key_chord_transition_distribution(
        key_chord_distribution,
        key_change_prob=key_change_prob,
        chord_change_prob=chord_change_prob)
    key_chord_loglik = np.log(key_chord_distribution)
    key_chord_transition_loglik = np.log(key_chord_transition_distribution)

    key_chords = _key_chord_viterbi(
        chord_frame_loglik, key_chord_loglik, key_chord_transition_loglik)

    # Add the inferred chord changes to the sequence, optionally adding key
    # signature(s) as well.
    current_key_name = None
    chord_rs = []
    key_rs = []
    for frame, (key, chord) in enumerate(key_chords):
        if chords_per_bar is not None:
            time = frame * seconds_per_chord
        else:
            print(f"chords_per_bar should not be None")
            return None, None

        if _PITCH_CLASS_NAMES[key] != current_key_name:
            key_rs.append(_PITCH_CLASS_NAMES[key])
            # A key change was inferred.
            current_key_name = _PITCH_CLASS_NAMES[key]
        if chord == NO_CHORD:
            figure = NO_CHORD
        else:
            root, kind = chord
            figure = '%s:%s' % (_PITCH_CLASS_NAMES[root], kind)
        chord_rs.append((figure, time))
    chord_items = []
    for chord1, chord2 in zip(chord_rs[:-1], chord_rs[1:]):
        pre_chord, pre_time = chord1
        _, back_time = chord2
        chord_items.append(Item(
            name="Chord",
            start=second2tick(pre_time, tempo=tempo),
            end=second2tick(back_time, tempo=tempo),
            value=pre_chord  # +"-m"
        ))
    chord_items.append(Item(
        name="Chord",
        start=second2tick(chord_rs[-1][1], tempo=tempo),
        end=sequence[-1].end,
        value=chord_rs[-1][0]
    ))
    return key_rs, chord_items
