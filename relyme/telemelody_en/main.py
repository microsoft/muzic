# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#


import re
from icecream import ic
from fairseq.models.transformer import TransformerModel
import miditoolkit
import os
import random
from tqdm import tqdm
from midi_utils import enc_vel, enc_ts, enc_tpo, encoding_to_midi, pos_resolution
from en_config import GEN_MODE, RHY_DEBUG, TEMPO
from en_constraintor import BeatConstraintor
from en_word_utils import get_in_word_pos, get_sents_form_mask, get_structure_mask

L2B_PREFIX = ""
T2N_PREFIX = ""
data_prefix = ""
prefix = ""

_PITCH_CLASS_NAMES = [
    'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

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
NO_CHORD = 'N.C.'

pitch_dict = dict()
for idx, name in enumerate(_PITCH_CLASS_NAMES):
    pitch_dict[name] = idx

Duration_vocab = dict([(float(x/100), 129+i)
                       for i, x in enumerate(list(range(25, 3325, 25)))])
MAX_DUR = int(max(Duration_vocab.values()))

def clean(word):
    word = re.sub('[ \xa0]+', '', word)
    word = re.sub('[,，] *', ',', word)
    word = re.sub('[。！？?] *', ".", word)
    word = re.sub('.{6} *', ".", word)
    word = re.sub('…+ *', ".", word)
    return word

SEP = '[sep]'
WORD = '[WORD]'

C2 = 36
C3 = 48
min_oct = 5
max_oct = 6


def adapt(pattern):
    num_dict = dict()
    prev = []
    for sent_idx, sent in enumerate(pattern):
        for sep_idx, sep in enumerate(sent):
            cur_len = len(sep)
            cur_starts = []
            if cur_len in num_dict and random.random() < 1.0:
                cur_starts = num_dict[cur_len]
                prev.extend(cur_starts)
                print('reuse rhythm:', cur_starts)
            else:
                if len(cur_starts) == 0:
                    offset = 0
                    if len(prev) and (sep[0] - prev[-1]) % 4 <= 1:
                        offset = sep[0] - prev[-1] + 2

                    cur_beats = (sep[0] - offset) % 4
                    new_sent = [cur_beats]
                    for item in sep[1:]:
                        if (item - offset - cur_beats) % 4 >= 2:
                            offset += (item - offset - cur_beats) % 4 - 1
                        if len(prev) >= 4 and len(set(prev[-4:])) == 1 and prev[-1] == (item - offset) % 4:
                            offset -= 1
                        new_sent.append((item - offset) % 4)
                        prev.append((item - offset) % 4)
                        cur_beats = new_sent[-1]

                    cur_starts = new_sent
                    num_dict[cur_len] = cur_starts
            pattern[sent_idx][sep_idx] = cur_starts
    return pattern

def adapt_e(e, align_idxs):
    tmp = [list(i) for i in e]
    last_pos = 0
    for i in range(len(tmp)):
        note = tmp[i]
        if note[3] <= min_oct * 12:
            note[3] = min_oct * 12 + note[3] % 12
        elif note[3] >= max_oct * 12 + 12:
            note[3] = max_oct * 12 + note[3] % 12

        # 16th note
        if note[1] % 2 == 1 and last_pos <= (16 * note[0] + note[1] - 1):
            note[1] -= 1
        if note[4] != 1 and (note[1] + note[4]) % 2 == 1:
            note[4] -= 1
        if last_pos >= 16 * note[0] + note[1]:
            tmp[i-1][4] -= last_pos - (16 * note[0] + note[1])
        last_pos = 16 * note[0] + note[1] + note[4]
        tmp[i] = note
    # ensure no rest in a word:
    words = []
    cur_word = []
    for idx, note in enumerate(tmp):
        if idx != 0 and idx in align_idxs:
            assert len(cur_word)
            if len(cur_word):
                words.append(cur_word)
                cur_word = []
        cur_word.append(note)
    if len(cur_word):
        words.append(cur_word)
    tmp = []
    for notes in words:
        first_note = notes[0]
        last_pos = 16 * first_note[0] + first_note[1] + first_note[4]
        tmp.append(first_note)
        for note in notes[1:]:
            note[0] = last_pos // 16
            note[1] = last_pos % 16
            tmp.append(note)
            last_pos += note[4]

    # remove empty bar:
    last_pos = 0
    offset = 0
    for note in tmp:
        cur_pos = 16 * (note[0] + offset) + note[1]
        while cur_pos - last_pos >= 16:
            offset -= 1
            cur_pos -= 16
        note[0] += offset
        last_pos = cur_pos + note[4]

    tmp = [tuple(i) for i in tmp]
    return tmp



if __name__ == '__main__':
    lyric2beats = TransformerModel.from_pretrained(
        f'checkpoints/{L2B_PREFIX}',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=f'data-bin/{L2B_PREFIX}',
    )
    trend2notes = TransformerModel.from_pretrained(
        f'checkpoints/{T2N_PREFIX}',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=f'data-bin/{T2N_PREFIX}'
    )

    os.makedirs(f'results/{prefix}/midi', exist_ok=True)
    
    with open(f'data/en/{data_prefix}/_lyric.txt', 'r') as f:
        lyr = f.readlines()
    with open(f'data/en/{data_prefix}/_chord.txt', 'r') as c:
        cho = c.readlines()
    with open(f'data/en/{data_prefix}/_syllables.txt', 'r') as s:
        syl = s.readlines()
    with open(f'data/en/{data_prefix}/_strct.txt', 'r') as st:
        strc = st.readlines()

    for song, sents in enumerate(tqdm(list(zip(lyr, cho, syl, strc)))):
    # for song, sents in enumerate(tqdm(list(zip(lyr, cho, syl)))):
        sents, bar_chords, syllables, strct = sents
        # ic(strct)
        # sents, bar_chords, syllables = sents
        sents = sents.strip()
        syllables = syllables.strip()
        tmp = bar_chords.split()
        bar_chords = []
        for item in tmp:
            if len(bar_chords) >= 2 and item == bar_chords[-1] and item == bar_chords[-2]:
                continue
            bar_chords.append(item)
        
        tmp = []
        cur_period = False
        cur_length = 0
        align_idxs = []
        for item in syllables.split():
            if item == SEP:
                if cur_length <= 0:
                    continue
                else:
                    cur_length = 0
                    if cur_period:
                        tmp.append('.')
                    else:
                        tmp.append(',')
                    cur_period = not cur_period
            else:
                if item[0] != '@':
                    align_idxs.append(
                        len([i for i in tmp if i not in [',', '.']]))
                tmp.append(item)
                cur_length += 1

        if tmp[-1] != '.':
            tmp[-1] = '.'

        word_num = len([i for i in tmp if i not in [',', '.']])

        beats = lyric2beats.translate(syllables,
                                    sampling=True,
                                    sampling_topk=2,
                                    temperature=0.5,
                                    beam=1,
                                    verbose=True,
                                    max_len_a=1,
                                    max_len_b=0,
                                    min_len=len(syllables.split()),
                                    )
        
        beats = beats.split()
        if GEN_MODE == "ReLyMe":
            sep_index = [i for i, b in enumerate(beats) if b == SEP]
            no_sep = [int(b) for b in beats if b != SEP]
            
            beatconstraintor = BeatConstraintor(sents, syllables, no_sep)
            _beats = beatconstraintor.rhythm_constraint()
            beats = [f"{b}" for b in _beats]
            for i in sep_index:
                beats.insert(i, SEP)
                
            if RHY_DEBUG:
                beatconstraintor.print_debug()
        
        beats_label = []
        cur_beats = []
        for item in beats:
            if item not in ['[sep]', WORD]:
                try:
                    cur_label = int(item)
                except BaseException as e:
                    if len(beats_label):
                        cur_label = beats_label[-1]
                    else:
                        cur_label = 0
                beats_label.append([cur_label])
            if len(beats_label) == word_num:
                break

        cur_idx = 0
        pattern = []
        cur_sent = []
        cur_sep = []
        word_idx = 0
        for word in tmp:
            if word not in [',', '.']:
                cur_sep.extend(beats_label[word_idx])
                cur_idx += len(beats_label[word_idx])
                word_idx += 1
            elif word == ',':
                if len(cur_sep):
                    cur_sent.append(cur_sep)
                    cur_sep = []
            elif word == '.':
                if len(cur_sep):
                    cur_sent.append(cur_sep)
                    cur_sep = []
                if len(cur_sent):
                    pattern.append(cur_sent)
                    cur_sent = []

        pattern = adapt(pattern)
        # ic(pattern)            
        mode = 'MAJ'

        bar_int = len(bar_chords)

        words = [mode]
        cur_bar = 0
        chords = []

        for sent_idx, sent in enumerate(pattern):
            for sect_idx, section in enumerate(sent):
                next_bar = False
                cur_chord = bar_chords[cur_bar % bar_int]
                print(cur_chord, end=' ')
                for idx, beat in enumerate(section):
                    if next_bar:
                        cur_bar += 1
                        cur_chord = bar_chords[cur_bar % bar_int]
                        print(cur_chord, end=' ')
                    next_bar = False
                    words.append(f'Chord_{cur_chord}')
                    chords.append(cur_chord)
                    if idx != len(section) - 1:
                        words.append('NOT')
                        if section[idx] > section[idx + 1]:
                            next_bar = True
                    elif sect_idx == len(sent) - 1:
                        words.append('AUT')
                    else:
                        words.append('HALF')
                    words.append(f'BEAT_{beat}')

                cur_bar += 1
        trend = ' '.join(words)
        ic(trend)
        def fix(items):
            tmp = []
            target_tokens = ['Bar', 'Pos', 'Pitch', 'Dur']
            i = 0
            for item in items:
                if item.split('_')[0] == target_tokens[i]:
                    tmp.append(item)
                    i = (i + 1) % len(target_tokens)
            return tmp
        
        structure, sents_form, in_word_pos = None, None, None
        if GEN_MODE == "ReLyMe":
            structure = get_structure_mask(tmp, strct)
            sents_form = get_sents_form_mask(tmp)
            in_word_pos = get_in_word_pos(syllables)
        
        notes = trend2notes.translate(  
                                        trend,
                                        sampling=True,
                                        sampling_topk=10,
                                        temperature=0.5,
                                        max_len_a=4 / 3,
                                        max_len_b=-4 / 3,
                                        min_len=(
                                            len(trend.split()) - 1) * 4 // 3,
                                        verbose=True,
                                        beam=1,
                                        strct=structure,
                                        in_word_pos=in_word_pos,
                                        sents_form=sents_form
                                    )
        # print(notes)
        enc = fix(notes.split())
        e = [ int(en.split('_')[1]) for en in enc ]
 
        # print(len(enc) // 4)
        e = [(e[i], e[i + 1], 0, e[i + 2], e[i + 3], enc_vel(127),
                enc_ts((4, 4)), enc_tpo(TEMPO)) for i in range(0, len(e) // 4 * 4, 4)]

        min_bar = min([i[0] for i in e])
        e = [tuple(k - min_bar if j == 0 else k for j,
                    k in enumerate(i)) for i in e]
        e.sort()
        e = e[:word_num]
        offset = 0
        e = [tuple(i) for i in e]
        e = adapt_e(e, align_idxs)
        # print(e)
        note_chords = []
        for chord, note in zip(chords, e):
            cur_idx = note[0] * 2
            if note[1] >= pos_resolution * 2:
                cur_idx += 1
            if len(note_chords) < cur_idx:
                note_chords = note_chords + \
                    [NO_CHORD] * (cur_idx - len(note_chords))
            if len(note_chords) == cur_idx:
                note_chords.append(chord)
            elif len(note_chords) == cur_idx + 1 and note_chords[-1] == NO_CHORD:
                note_chords[-1] = chord

        for i in range(1, len(note_chords)):
            if note_chords[i] == NO_CHORD:
                note_chords[i] = note_chords[i-1]

        midi_obj = encoding_to_midi(e)
        midi_obj.tempo_changes[0].tempo = TEMPO
        midi_obj.instruments[0].notes.sort(key=lambda x: (x.start, -x.end))

        ticks = midi_obj.ticks_per_beat
        midi_obj.instruments[0].name = 'melody'
        midi_obj.instruments.append(miditoolkit.Instrument(
            program=0, is_drum=False, name='chord'))  # piano
        midi_obj.instruments[0].program = 40  # violin
        midi_obj.instruments[1].notes = []
        tmp = []
        cur_period = False
        cur_length = 0
        for item in sents.strip().split():
            if item == SEP:
                if cur_length <= 0:
                    continue
                else:
                    cur_length = 0
                    if cur_period:
                        tmp.append('.')
                    else:
                        tmp.append(',')
                    cur_period = not cur_period
            else:
                tmp.append(item.lower())
                cur_length += 1

        if tmp[-1] != '.':
            tmp[-1] = '.'
        lyrics = []
        for word in tmp:
            if word not in [',', '.']:
                lyrics.append(word)
            else:
                lyrics[-1] += word

        note_idx = 0

        word_idx = 0
        for idx, word in enumerate(lyrics):
            if word not in [',', '.']:
                note = midi_obj.instruments[0].notes[align_idxs[word_idx]]
                midi_obj.lyrics.append(
                    miditoolkit.Lyric(text=word, time=note.start))
                word_idx += 1
            else:
                midi_obj.lyrics[-1].text += word
        for idx, chord in enumerate(note_chords):
            if chord != NO_CHORD:
                root, type = chord.split(':')
                root = pitch_dict[root]
                midi_obj.instruments[1].notes.append(
                    miditoolkit.Note(velocity=80, pitch=C2 + root, start=(idx * 2) * ticks, end=(idx * 2 + 2) * ticks))
                for shift in _CHORD_KIND_PITCHES[type]:
                    midi_obj.instruments[1].notes.append(
                        miditoolkit.Note(velocity=80, pitch=C3 + (root + shift) % 12, start=(idx * 2) * ticks, end=(idx * 2 + 2) * ticks))

        midi_obj.dump(f'results/{prefix}/{GEN_MODE}/{song}_{GEN_MODE}.mid')
