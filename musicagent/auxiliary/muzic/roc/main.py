import sqlite3
import random
import copy
import numpy as np
import argparse
import re
from math import ceil
from utils.lyrics_match import Lyrics_match
from midiutil.MidiFile import MIDIFile
from fairseq.models.transformer_lm import TransformerLanguageModel
from cnsenti import Sentiment
import miditoolkit
from midi2dict import midi_to_lyrics
from textblob import TextBlob

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

# custom_lm = TransformerLanguageModel.from_pretrained('music-ckps/', 'checkpoint_best.pt', tokenizer='space',
#                                                      batch_size=8192).cuda()


def select_melody(c, is_maj, is_chorus, length, last_bar, chord, chord_ptr, is_last_sentence):
    cursor = c.execute(
        "SELECT DISTINCT NOTES, CHORDS from MELOLIB  where LENGTH = '{}' and CHORUS = '{}' and MAJOR = '{}' ".format(
            length, is_chorus, is_maj))  # and MAJOR = '{}'
    candidates_bars = []
    if is_debug:
        print("Retrive melody...")
    for row in cursor:
        notes = row[0]
        cd_ = row[1]
        candidates_bars.append((notes, cd_))

    # Filter by chords.
    chord_list_ = chord.strip().split(' ')
    chord_list_ = chord_list_[chord_ptr:] + chord_list_[:chord_ptr]
    re_str = ''

    if not is_last_sentence:
        key = ''
    else:
        if is_maj:
            key = ' C:'
        else:
            key = ' A:m'

    # For the given chord progression, we generate a regex like:
    # A:m F: G: C: -> ^A:m( A:m)*( F:)+( G:)+( C:)*$|^A:m( A:m)*( F:)+( G:)*$|^A:m( A:m)*( F:)*$|^A:m( A:m)*$
    # Given the regex, we find matched pieces.
    # We design the regex like this because alternations in regular expressions are evaluated from left to right,
    # the piece with the most various chords will be selected, if there's any.
    for j in range(len(chord_list_), 0, -1):
        re_str += '^({}( {})*'.format(chord_list_[0], chord_list_[0])
        for idx in range(1, j):
            re_str += '( {})+'.format(chord_list_[idx])
        re_str = re_str[:-1]
        re_str += '*{})$|'.format(key)
    re_str = re_str[:-1]

    tmp_candidates = []
    for row in candidates_bars:
        if re.match(r'{}'.format(re_str), row[1]):
            tmp_candidates.append(row)

    if len(tmp_candidates) == 0:
        re_str = '^{}( {})*$'.format(chord_list_[-1], chord_list_[-1])
        for row in candidates_bars:
            if re.match(r'{}'.format(re_str), row[1]):
                tmp_candidates.append(row)

    if len(tmp_candidates) > 0:
        candidates_bars = tmp_candidates
    else:
        if is_maj:
            re_str = '^C:( C:)*$'
        else:
            re_str = '^A:m( A:m)*$'
        for row in candidates_bars:
            if re.match(r'{}'.format(re_str), row[1]):
                tmp_candidates.append(row)
        if len(tmp_candidates) > 0:
            candidates_bars = tmp_candidates

    candidates_cnt = len(candidates_bars)
    if candidates_cnt == 0:
        if is_debug:
            print('No Matched Rhythm as {}'.format(length))
        return []

    if last_bar == None:  # we are at the begining of a song, random select bars.
        if is_debug:
            print('Start a song...')

        def not_too_high(bar):
            notes = bar.split(' ')[:-1][3::5]
            notes = [int(x[6:]) for x in notes]
            for i in notes:
                if 57 > i or i > 66:
                    return False
            return True

        tmp = []
        for bar in candidates_bars:
            if not_too_high(bar[0]):
                tmp.append(bar)
        return tmp
    else:
        last_note = int(last_bar.split(' ')[-3][6:])
        # tendency
        selected_bars = []
        prefer_note = None

        # Major C
        if is_maj:
            if last_note % 12 == 2 or last_note % 12 == 9:
                prefer_note = last_note - 2
            elif last_note % 12 == 5:
                prefer_note = last_note - 1
            elif last_note % 12 == 11:
                prefer_note = last_note + 1
                # Minor A
        else:
            if last_note % 12 == 11 or last_note % 12 == 2:  # 2 -> 1, 4 -> 3
                prefer_note = last_note - 2
            elif last_note % 12 == 6:  # 6 -> 5
                prefer_note = last_note - 1
            elif last_note % 12 == 7:  # 7-> 1
                prefer_note = last_note + 2

        if prefer_note is not None:
            for x in candidates_bars:
                if x[0][0] == prefer_note:
                    selected_bars.append(x)
        if len(selected_bars) > 0:
            if is_debug:
                print('Filter by tendency...')
            candidates_bars = selected_bars

        selected_bars = []
        for bar in candidates_bars:
            first_pitch = int(bar[0].split(' ')[3][6:])
            if (first_pitch > last_note - 8 and first_pitch < last_note + 8):
                selected_bars.append(bar)
        if len(selected_bars) > 0:
            if is_debug:
                print('Filter by pitch range...')
            return selected_bars

    # No candidates yet? randomly return some.
    if is_debug:
        print("Randomly selected...")
    return candidates_bars


def lm_score(custom_lm, bars, note_string, bar_idx):
    tmp_string = []
    n = ' '.join(note_string.split(' ')[-100:])
    for sbar in bars:
        sbar_, _ = fill_template(sbar[0], bar_idx)
        tmp_string.append(n + sbar_)

    score = [x['score'].item() for x in custom_lm.score(tmp_string)]

    assert len(score) == len(tmp_string)
    tmp = list(zip(bars, score))

    tmp.sort(key=lambda x: x[1], reverse=True)

    tmp = tmp[:30]
    best_score = tmp[0][1]
    res = []
    for x in tmp:
        if best_score - x[1] < 0.1:
            res.append(x[0])
    return res


def get_chorus(chorus_start, chorus_length, lyrics):
    return range(chorus_start, chorus_start + chorus_length)


def save_demo(notes_str, select_chords, name, lang, sentence, word_counter):
    pitch_dict = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
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

    print('Save the melody to {}.mid'.format(name))

    mf = MIDIFile(2)  # only 1 track
    melody_track = 0  # the only track
    chord_track = 1
    time = 0  # start at the beginning
    channel = 0

    mf.addTrackName(melody_track, time, "melody")
    mf.addTrackName(chord_track, time, "chord")
    mf.addTempo(melody_track, time, 120)
    mf.addTempo(chord_track, time, 120)

    notes = notes_str.split(' ')

    cnt = 0
    sen_idx = 0

    chord_time = []
    for i in range(len(notes) // 5):
        if is_debug:
            print('wirting idx: ', i)
        # cadence = notes[5 * i]
        bar = int(notes[5 * i + 1][4:])
        pos = int(notes[5 * i + 2][4:])  # // pos_resolution
        pitch = int(notes[5 * i + 3][6:])
        dur = int(notes[5 * i + 4][4:]) / 4

        time = bar * 4 + pos / 4  # + delta

        # if cadence == 'HALF':
        #     delta += 2
        # if cadence == 'AUT':
        #    delta += 4

        mf.addNote(melody_track, channel, pitch, time, dur, 100)

        # fill all chords into bars before writing notes
        if cnt == 0:
            cds = select_chords[sen_idx].split(' ')
            t = time - time % 2
            if len(chord_time) > 0:
                blank_dur = t - chord_time[-1] - 2
                insert_num = int(blank_dur / 2)

                if is_debug:
                    print('Chords:', cds[0].split(':'))
                root, cd_type = cds[0].split(':')
                root = pitch_dict[root]
                for i in range(insert_num):
                    for shift in _CHORD_KIND_PITCHES[cd_type]:
                        mf.addNote(chord_track, channel, 36 + root + shift, chord_time[-1] + 2, 2, 75)
                    chord_time.append(chord_time[-1] + 2)

            if is_debug:
                print('begin sentence:', sen_idx)
            for cd in cds:
                root, cd_type = cd.split(':')
                root = pitch_dict[root]
                # mf.addNote(chord_track, channel, 36+root, t, 2, 75)  # 36 is C3
                for shift in _CHORD_KIND_PITCHES[cd_type]:
                    mf.addNote(chord_track, channel, 36 + root + shift, t, 2, 75)
                chord_time.append(t)
                t += 2

        cnt += 1
        if cnt == word_counter[sen_idx]:
            cnt = 0
            sen_idx += 1
    name += '.mid'
    with open(name, 'wb') as outf:
        mf.writeFile(outf)

    midi_obj = miditoolkit.midi.parser.MidiFile(name)
    if lang == 'zh':
        lyrics = ''.join(sentence)
    else:
        print(sentence)
        lyrics = ' '.join(sentence).split(' ')
        print(lyrics)
    word_idx = 0
    for idx, word in enumerate(lyrics):
        if word not in [',', '.', '']:
            note = midi_obj.instruments[0].notes[word_idx]
            midi_obj.lyrics.append(
                miditoolkit.Lyric(text=word, time=note.start))
            word_idx += 1
        else:
            midi_obj.lyrics[-1].text += word
    # print(midi_obj.lyrics)
    midi_obj.dump(f'{name}', charset='utf-8')
    return midi_to_lyrics(midi_obj)


def fill_template(s_bar, bar_idx):
    notes = s_bar.split(' ')
    tmp = []
    last_bar_idx = notes[1][4:]
    for i in range(len(notes)):
        if i % 5 == 1:
            if notes[i][4:] != last_bar_idx:
                bar_idx += 1
                last_bar_idx = notes[i][4:]
            tmp.append('bar_' + str(bar_idx))
        else:
            tmp.append(notes[i])
    return ' '.join(tmp), bar_idx + 1


def splice(bar1, bar2):
    """
        Cancatenate bar1 and bar2
        In bar1, bar index is replaced while in bar2 X or Y is remained like 'X {} {} {} {} '.format(pos, pitch,dur,cadence)
    """
    if bar1 == '':
        return bar2
    if bar2 == '':
        return bar1

    assert bar1[-1] == ' '  # For the ease of concatenation, there's a space at the end of bar
    assert bar2[-1] == ' '
    notes1 = bar1.split(' ')[:-1]
    notes2 = bar2.split(' ')[:-1]
    bar_cnt = len(set(notes1[1::5]))

    # If the last note ending time in bar1 is not far from the begining time of the first note in bar2, just return bar1 + bar2
    # Calculate the note intervals in bars. If interval between two bars <= the average interval inside a bar, then it is regarded as 'not far away'.
    def get_interval(notes):
        begin = []
        dur = []

        if notes[1][4:] != 'X' and notes[1][4:] != 'Y':
            start_bar = int(notes[1][4:])
        else:
            start_bar = 0

        for idx in range(len(notes) // 5):
            if notes[5 * idx + 1][4:] == 'X':
                bar_idx_ = 0
            elif notes[5 * idx + 1][4:] == 'Y':
                bar_idx_ = 1
            else:
                bar_idx_ = int(notes[5 * idx + 1][4:])
            begin.append(16 * (bar_idx_ - start_bar) + int(notes[5 * idx + 2][4:]))
            dur.append(int(notes[5 * idx + 4][4:]))
        end = list(np.array(begin) + np.array(dur))

        return list(np.array(begin[1:]) - np.array(end[:-1])), begin[0], end[-1] - 16 if end[-1] > 16 else end[-1]

    inter1, _, end1 = get_interval(notes1)
    inter2, begin2, _ = get_interval(notes2)

    def avg(notes):
        return sum(notes) / len(notes)

    avg_interval = avg(inter1 + inter2)

    last_bar1_idx = int(notes1[-4][4:])
    bar2, _ = fill_template(bar2, last_bar1_idx + 1)

    if avg_interval < (16 - end1 + begin2):
        # If interval between two bars is big, shift the second bar forward.
        notes2 = bar2.split(' ')[:-1]
        tmp = ''
        for idx in range(len(notes2) // 5):
            pos = int(notes2[5 * idx + 2][4:]) - (16 - end1 + begin2)
            bar_idx_ = int(notes2[5 * idx + 1][4:])
            if pos < 0:
                bar_idx_ += pos // 16
                pos = pos % 16
            tmp += '{} bar_{} Pos_{} {} {} '.format(notes2[5 * idx], bar_idx_, pos, notes2[5 * idx + 3],
                                                    notes2[5 * idx + 4])

        return bar1 + tmp
    else:

        return bar1 + bar2


def not_mono(bar):
    """
        Filter monotonous pieces.
    """
    notes = bar.split(' ')[:-1][3::5]
    notes = [int(x[6:]) for x in notes]

    tmp = [0] * 128
    for idx in range(len(notes)):
        tmp[int(notes[idx])] = 1
    if (1 < len(notes) <= 3 and sum(tmp) == 1) or (len(notes) >= 4 and sum(tmp) < 3):
        return False
    return True


def not_duplicate(bar1, bar2):
    """
        De-duplication, only care about the pitch.
    """
    notes1 = bar1.split(' ')[:-1][3::5]  # For the ease of concatenation, there's a space at the end of bar
    notes2 = bar2.split(' ')[:-1][3::5]
    return notes1 != notes2


def no_keep_trend(bars):
    def is_sorted(a):
        return all([a[i] <= a[i + 1] for i in range(len(a) - 1)])

    candidates_bars = []
    for bar_and_chord in bars:
        bar = bar_and_chord[0]
        notes = bar.split(' ')[:-1][3::5]
        notes = [int(x[6:]) for x in notes]
        if not is_sorted(notes):
            candidates_bars.append(bar_and_chord)
    return candidates_bars


def polish(bar, last_note_end, iscopy=False):
    """
        Three fuctions:
        1. Avoid bars overlapping.
        2. Make the first note in all bars start at the position 0.
        3. Remove rest and cadence in a bar.
    """
    notes = bar.strip().split(' ')
    tmp = ''
    first_note_start = 0
    is_tuned = False
    for idx in range(len(notes) // 5):
        pos = int(notes[5 * idx + 2][4:])
        bar_idx_ = int(notes[5 * idx + 1][4:])
        dur = int(notes[5 * idx + 4][4:])

        this_note_start = 16 * bar_idx_ + pos

        cadence = 'NOT'

        if idx == 0:
            first_note_start = this_note_start
            blank_after_last_note = 16 - last_note_end % 16
            threshold = blank_after_last_note
        else:
            threshold = 0

        if dur == 1:  # the minimum granularity is a 1/8 note.
            dur = 2
        if dur > 8:  # the maximum granularity is a 1/2 note.
            dur = 8

        # Function 3:
        if this_note_start - last_note_end != threshold:
            pos += (last_note_end + threshold - this_note_start)
            bar_idx_ += pos // 16
            pos = pos % 16

        if idx == len(notes) // 5 - 2:
            if 12 < pos + dur <= 16 or len(notes) // 5 <= 4:
                dur = 16 - pos
                is_tuned = True
        if idx == len(notes) // 5 - 1:
            if is_tuned:
                pos = 0
            else:
                if 12 < pos + dur <= 16:
                    dur += 6
            cadence = 'HALF'  # just for the ease of model scoring

        last_note_end = 16 * bar_idx_ + pos + dur

        assert pos <= 16
        tmp += '{} bar_{} Pos_{} {} Dur_{} '.format(cadence, bar_idx_, pos, notes[5 * idx + 3], dur)

    return tmp, bar_idx_ + 1, last_note_end


def chord_truc(bar, schord):
    """
        Given a bar string, remove redundant chords.
    """
    schord_list = schord.split(' ')
    notes = bar.strip().split(' ')
    start_pos = 16 * int(notes[1][4:]) + int(notes[2][4:])
    end_pos = 16 * int(notes[-4][4:]) + int(notes[-3][4:]) + int(notes[-1][4:])
    duration = end_pos - start_pos
    chord_num = ceil(duration / 8)
    assert chord_num >= 1, 'bar:{},chord:{}'.format(bar, schord)
    if len(schord_list) >= chord_num:
        schord_list = schord_list[:chord_num]
    else:
        tmp = []
        for i in schord_list:
            tmp.append(i)
            tmp.append(i)
        schord_list = tmp[:chord_num]
    return schord_list


def polish_chord(bar, schord, chord, chord_ptr):
    """
        Align chords and the bar. When this function is called, the bar index is already replaced by the true index instead of X or Y.
        In our setting, there's 2 chords in a bar. Therefore for any position % 8==0, we write a chord.
        Of course, you can modify this setting as needed.
    """
    schord_list = chord_truc(bar, schord)
    last_chord = schord_list[-1]
    schord = ' '.join(schord_list)

    chord_list = chord.split(' ')
    if last_chord not in chord_list:
        chord_ptr = (chord_ptr + 1) % len(chord_list)
    else:
        chord_ptr = (chord_list.index(last_chord) + 1) % len(chord_list)
    return schord, chord_ptr


# if __name__ == '__main__':
def main(
        custom_lm, 
        lyrics_corpus=['明月几时有 把酒问青天 不知天上宫阙 今夕是何年'],
        chord_corpus=['zh C: G: A:m E: D:m G:'],
        output_file_name='generated',
        db_path='database/ROC.db'
):
    use_sentiment = False
    global is_debug
    is_debug = False
    conn = sqlite3.connect(db_path)
    # global c
    c = conn.cursor()
    print("Database connected")

    for lyrics, chord in zip(lyrics_corpus, chord_corpus):

        lang = chord[:2]
        assert lang in ['zh', 'en']  # Note that ROC is not language-sensitive, you can extend this.

        chord = chord[2:].strip()
        print('CHORD:', chord)

        chord_ptr = 0

        is_maj = 1

        if lang == 'zh':
            sentence = lyrics.strip().split(' ')  # The list of lyric sentences
            name = sentence[0]

            if use_sentiment:
                senti = Sentiment()
                pos = 0
                neg = 0
                for s in sentence:
                    result = senti.sentiment_calculate(s)
                    pos += result['pos']
                    neg += result['neg']
                if neg < 0 and pos >= 0:
                    is_maj = 1
                elif pos < 0 and neg >= 0:
                    is_maj = 0
                else:
                    if pos / neg < 1:
                        is_maj = 0
                    else:
                        is_maj = 1

        elif lang == 'en':
            sentence = lyrics.strip().split('[sep]')
            name = sentence[0]
            sentence = [len(x.strip().split(' ')) * '_' for x in sentence]

            if use_sentiment:
                sent = '.'.join(sentence)
                blob = TextBlob(sent)
                polarity = 0
                for s in blob.sentences:
                    polarity += s.sentiment.polarity
                if polarity >= 0:
                    is_maj = 1
                else:
                    is_maj = 0

        print('Tonality:', is_maj)

        # structure recognition
        parent, chorus_start, chorus_length = Lyrics_match(
            sentence)  # The last element must be -1, because the chord should go back to tonic
        if is_debug:
            print('Struct Array: ', parent)

        chorus_range = get_chorus(chorus_start, chorus_length, lyrics)
        if is_debug:
            print('Recognized Chorus: ', chorus_start, chorus_length)

        select_notes = []  # selected_melodies
        select_chords = []  # selected chords
        is_chorus = 0  # is a chorus?
        note_string = ''  # the 'melody context' mentioned in the paper.
        bar_idx = 0  # current bar index. it is used to replace bar index in retrieved pieces.
        last_note_end = -16
        # is_1smn = 0             # Does 1 Syllable align with Multi Notes? In the future, we will explore better methods to realize this function. Here by default, we disable it.

        for i in range(len(sentence)):
            if lang == 'zh':
                print('Lyrics: ', sentence[i])
            else:
                print('Lyrics: ', lyrics.strip().split('[sep]')[i])

            is_last_sentence = (i == len(sentence) - 1)

            if i in chorus_range:
                is_chorus = 1
            else:
                is_chorus = 0

            cnt = len(sentence[i])
            if cnt <= 2 and parent[i] == -2:  # if length is too short, do not partially share
                parent[i] = -1

            # Following codes correspond to 'Retrieval and Re-ranking' in Section 3.2.
            # parent[i] is the 'struct value' in the paper.
            if parent[i] == -1:
                if is_debug:
                    print('No sharing.')
                # one_syllable_multi_notes_probabilty = random.randint(1,100)
                # if one_syllable_multi_notes_probabilty == 1:
                #     is_1smn = 1
                #     connect_notes = random.randint(1,2)
                #     cnt += connect_notes
                #     connect_start = random.randint(1,cnt)
                #     print('One Syllable Multi Notes range:',connect_start, connect_start + connect_notes)

                if len(select_notes) == 0:  # The first sentence of a song.
                    last_bar = None
                else:
                    last_bar = select_notes[-1]

                selected_bars = select_melody(c, is_maj, is_chorus, cnt, last_bar, chord, chord_ptr, is_last_sentence)
                if cnt < 9 and len(selected_bars) > 0:
                    selected_bars = lm_score(custom_lm, selected_bars, note_string, bar_idx)
                    # selected_bars = no_keep_trend(selected_bars)
                    bar_chord = selected_bars[random.randint(0, len(selected_bars) - 1)]
                    s_bar = bar_chord[0]
                    s_chord = bar_chord[1]
                    s_bar, bar_idx = fill_template(s_bar,
                                                    bar_idx)  # The returned bar index is the first bar index which should be in the next sentence, that is s_bar + 1.
                else:  # If no pieces is retrieved or there are too many syllables in a lyric.
                    if is_debug:
                        print('No pieces is retrieved or there are too many syllables in a lyric. Split the lyric.')
                    s_bar = ''
                    s_chord = ''
                    origin_cnt = cnt
                    error = 0
                    while cnt > 0:
                        l = max(origin_cnt // 3, 5)
                        r = max(origin_cnt // 2, 7)  # Better to use long pieces, for better coherency.
                        split_len = random.randint(l, r)
                        if split_len > cnt:
                            split_len = cnt
                        if is_debug:
                            print('Split at ', split_len)
                        selected_bars = select_melody(c, is_maj, is_chorus, split_len, last_bar, chord, chord_ptr,
                                                        is_last_sentence)
                        if len(selected_bars) > 0:
                            selected_bars = lm_score(custom_lm, selected_bars, note_string + s_bar, bar_idx)
                            bar_chord = selected_bars[random.randint(0, len(selected_bars) - 1)]
                            last_bar = bar_chord[0]
                            last_chord = bar_chord[1]
                            s_bar = splice(s_bar, last_bar)
                            s_chord += ' ' + last_chord

                            # Explanation: if this condition is true, i.e., the length of s_bar + last_bar == the length of last_bar,
                            # then the only possibility is that we are in the first step of this while loop. We need to replace the bar index in retrieved pieces with the true bar index.
                            # In the following steps, there is no need to do so because there is a implicit 'fill_template' in 'splice'.
                            if len(s_bar) == len(last_bar):
                                s_bar, bar_idx = fill_template(s_bar, bar_idx)
                            s_chord, chord_ptr = polish_chord(s_bar, s_chord, chord, chord_ptr)
                            last_bar = s_bar
                            cnt -= split_len
                        else:
                            error += 1
                            if error >= 10:
                                print('Database has not enough pieces to support ROC.')
                                exit()

                    s_chord = s_chord[1:]

                s_bar, bar_idx, last_note_end = polish(s_bar, last_note_end)
                s_chord, chord_ptr = polish_chord(s_bar, s_chord, chord, chord_ptr)
                note_string += s_bar
                select_notes.append(s_bar)
                select_chords.append(s_chord)
                if is_debug:
                    print('Selected notes: ', s_bar)
                    print('Chords: ', s_chord)

            elif parent[i] == -2:
                if is_debug:
                    print('Share partial melody from the previous lyric.')

                l = min(cnt // 3,
                        3)  # As mentioned in 'Concatenation and Polish' Section, for adjacents lyrics having the same syllabels number,
                r = min(cnt // 2, 5)  # we 'polish their melodies to sound similar'

                # modify some notes then share.
                replace_len = random.randint(l, r)
                last_bar = ' '.join(select_notes[-1].split(' ')[:- replace_len * 5 - 1]) + ' '
                tail = select_notes[-1].split(' ')[- replace_len * 5 - 1:]
                last_chord = ' '.join(chord_truc(last_bar, select_chords[-1]))
                selected_bars = select_melody(c, is_maj, is_chorus, replace_len, last_bar, chord, chord_ptr,
                                                is_last_sentence)
                selected_bars = lm_score(custom_lm, selected_bars, note_string + last_bar, bar_idx)
                for bar_chord in selected_bars:
                    bar = bar_chord[0]
                    s_chord = bar_chord[1]
                    s_bar = splice(last_bar, bar)
                    if not_mono(s_bar) and not_duplicate(s_bar, select_notes[-1]):
                        s_chord = last_chord + ' ' + s_chord
                        break

                s_bar, bar_idx = fill_template(s_bar, bar_idx)
                s_bar = s_bar.split(' ')

                for i in range(2, len(tail)):  # Modify duration
                    if i % 5 == 2 or i % 5 == 1:  # dur and cadence
                        s_bar[-i] = tail[-i]
                s_bar = ' '.join(s_bar)

                s_bar, bar_idx, last_note_end = polish(s_bar, last_note_end, True)
                s_chord, chord_ptr = polish_chord(s_bar, s_chord, chord, chord_ptr)
                note_string += s_bar
                select_notes.append(s_bar)
                select_chords.append(s_chord)

                if is_debug:
                    print('Modified notes: ', s_bar)
                    print('chords: ', s_chord)
            else:
                # 'struct value is postive' as mentioned in the paper, we directly share melodies.
                if is_debug:
                    print('Share notes with sentence No.', parent[i])

                s_bar = copy.deepcopy(select_notes[parent[i]])
                s_chord = copy.deepcopy(select_chords[parent[i]])
                s_bar, bar_idx = fill_template(s_bar, bar_idx)
                s_bar, bar_idx, last_note_end = polish(s_bar, last_note_end, True)
                s_chord, chord_ptr = polish_chord(s_bar, s_chord, chord, chord_ptr)
                note_string += s_bar
                select_notes.append(s_bar)
                select_chords.append(s_chord)

            if is_debug:
                print(
                    '----------------------------------------------------------------------------------------------------------')

        if is_debug:
            print(select_chords)
            print(select_notes)

        output = save_demo(note_string, select_chords, output_file_name, lang,
                    lyrics.strip().split('[sep]') if lang == 'en' else sentence, [len(i) for i in sentence])

        print(output)
        print(
            '--------------------------------------------A song is composed.--------------------------------------------')
    conn.close()
    return output

if __name__ == '__main__':
    model = TransformerLanguageModel.from_pretrained('music-ckps/', 'checkpoint_best.pt', tokenizer='space',
                                                     batch_size=8192).cuda()
    main(model)

