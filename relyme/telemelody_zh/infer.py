"""
Infer Module
"""
import miditoolkit
from icecream import ic
from fairseq.models.transformer import TransformerModel
from utils import enc_vel, enc_ts, enc_tpo, encoding_to_midi, pos_resolution
from config import TEMPO

L2B_PREFIX = ""
T2N_PREFIX = ""

_PITCH_CLASS_NAMES = [
    'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'
]

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

def stage_1_get_l2b(lyrics):
    """Get Lyrics2Beat results
    Args:
        lyrics (string): lyrics

    Returns:
        start_label (list): start beat of each word, range from 0 to 3
    """
    needed_len = len([ l for l in lyrics if l != '.' ])

    pred_cnt = 0
    while pred_cnt < 30:
        pred = lyric2beats.translate(
            lyrics,
            sampling=True,
            sampling_topk=2,
            temperature=0.5,
            beam=1,
            verbose=True,
            max_len_a=1,
            max_len_b=0,
            min_len=len(lyrics.split()),
        )

        start_label = [ int(i) for i in pred.tolist()[0] if i not in ['[sep]'] ]
        pred_cnt += 1
        print(pred_cnt)
        if len(start_label) >= needed_len:
            break
        if pred_cnt == 29:
            raise ValueError("Takes too many iterations to generate L2B results")

    return start_label

def adapt_b(beats):
    """Adjust beats, to (1) reuse beat pattern, (2) reduce the time interval
    """
    new_beats = []
    pattern_dict = dict()
    for pat in beats:
        cur_len = len(pat)
        if cur_len in pattern_dict:
            new_patt = pattern_dict[cur_len]
        else:
            offset = 0
            cur_beat = pat[0]
            new_patt = [cur_beat]
            for nxt_beat in pat[1:]:
                interval = (nxt_beat - offset - cur_beat) % 4
                if interval >= 2:
                    offset += interval - 1

                new_patt.append((nxt_beat - offset) % 4)
                cur_beat = new_patt[-1]

            pattern_dict[cur_len] = new_patt
        
        new_beats.append(new_patt)
    return new_beats

def stage_2_get_b2t(lyrics, start_label, chord_pro):
    """Get Beat2Trend

    Args:
        lyrics (string): lyrics
        start_label (list): start beat of each word, range from 0 to 3
        chord_pro (list): chord progression

    Returns:
        trend (list): (Beat, Chord, [Aut, Half, Not])
        chords (list): chords of each bar
    """

    # get lyrics pattern
    l_idx = 0
    pattern = []  # split by '.', group L2B outputs into sentence pattern
    cur_sent = [] # split by ','
    tmp_sent = []
    for l in lyrics:
        if l == ',':
            cur_sent.append(tmp_sent)
            tmp_sent = []

        elif l in ['.', '?']:
            cur_sent.append(tmp_sent)
            pattern.append(cur_sent)
            cur_sent = []
            tmp_sent = []

        else:
            tmp_sent.append(start_label[l_idx])
            l_idx += 1
    # sent_pattern = adapt_b(sum(pattern, []))
    
    # para_num = [ len(pat) for pat in pattern ]
    # offset = 0
    # new_pattern = []

    # # ic(sent_pattern, para_num)
    # for n in para_num:
    #     new_pattern.append(list(sent_pattern[offset:offset+n]))
    #     offset += n
    
    # pattern = new_pattern
    
    word = ["MAJ"]
    chords = []
    cur_bar = 0
    bar_num = len(chord_pro)
    for sent in pattern:
        for sect_idx, section in enumerate(sent):
            next_bar = False
            cur_chord = chord_pro[ cur_bar % bar_num ]
            for idx, beat in enumerate(section):
                if next_bar:
                    cur_bar += 1
                    cur_chord = chord_pro[ cur_bar % bar_num ]
                next_bar = False
                word.append(f'Chord_{cur_chord}')
                chords.append(cur_chord)

                if idx != len(section) - 1:
                    word.append('NOT')
                    if section[idx] > section[idx + 1] or \
                        (idx > 0 and section[idx + 1] == section[idx - 1] == section[idx]):
                        next_bar = True
                elif sect_idx == len(sent) - 1:
                    word.append('AUT')
                else:
                    word.append('HALF')

                word.append(f'BEAT_{beat}')
            cur_bar += 1

    trend = ' '.join(word)
    return trend, chords

trend2notes = TransformerModel.from_pretrained(
    'checkpoints/light_lmd_matched_16-bin',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data-bin/lmd_matched_16-bin',
    max_target_positions=2048
)
def stage_3_get_t2m(trend, _lyrics, sent_info):
    """Get Treneds2Melody results

    Args:
        trend (string): (Beat, Chord, [Aut, Half, Not])
        lyrics (string): lyrics, to perform pitch constraints
        letter_pos (list): letter position to perform phrase constraints

    Returns:
        [type]: [description]
    """
    notes = trend2notes.translate(
                            trend,
                            lyrics=_lyrics,
                            letter_pos=sent_info,
                            sampling=True,
                            sampling_topk=10,
                            temperature=0.5,
                            max_len_a=4 / 3,
                            max_len_b=0,
                            min_len=(len(trend.split()) - 1) * 4 // 3,
                            beam=1,
                            print_alignment=True,
                            )
    return notes

def adapt_e(e):
    tmp = [ list(t) for t in e ]
    prev_bar, prev_pos, prev_end, prev_dur = 0, 0, 0, 0
    for idx, note in enumerate(tmp):
        # Bar
        if note[0] - prev_bar > 1:
            note[0] = (prev_bar + 1) if note[1] < prev_pos else prev_bar

        # Pos
        if note[1] % 2 == 1 and prev_end <= (16 * note[0] + note[1] - 1): # 16th note
            note[1] -= 1
        if note[4] != 1 and (note[1] + note[4]) % 2 == 1:
            note[4] -= 1

        # Pitch
        if note[3] < 60: # if lower than C4 
            note[3] += 12        

        # Dur
        curr_pos = 16 * note[0] + note[1]
        if prev_end > curr_pos:
            tmp[idx-1][4] = curr_pos - (16 * prev_bar + prev_pos)

        prev_bar, prev_pos, prev_dur = note[0], note[1], note[4]
        prev_end = 16 * prev_bar + prev_pos + prev_dur

    tmp = [tuple(i) for i in tmp]
    return tmp

C2 = 36
C3 = 48
pitch_dict = {name: idx for idx, name in enumerate(_PITCH_CLASS_NAMES)}
def stage_4_get_m2m(notes, chords, _lyrics, midi_filename) -> None:
    """Melody2Midi

    Args:
        notes (string): from stage3 t2m
        chords (string): from stage2 b2t, chord of each bar
        _lyrics (string): lyrics
        midi_filename (string): filename of midi to be created
    """
    # eliminate diordered notes info
    i = 0
    ordered_notes = []
    target_tokens = ['Bar', 'Pos', 'Pitch', 'Dur']
    for item in notes.split():
        item = item.split('_')
        flag, num = item[0], int(item[1])
        if flag == target_tokens[i]:
            ordered_notes.append(num)
            i = (i + 1) % len(target_tokens)

    # encoding to midi encoding
    notes_num = len(ordered_notes) // 4 * 4
    encoding = [
        (
            ordered_notes[i],
            ordered_notes[i + 1],
            0,
            ordered_notes[i + 2],
            ordered_notes[i + 3],
            enc_vel(127),
            enc_ts((4, 4)),
            enc_tpo(TEMPO)
        )
        for i in range(0, notes_num, 4)
    ]

    min_bar = min([i[0] for i in encoding])
    encoding = [ tuple(k - min_bar for k in i) for i in encoding ]
    encoding.sort()
    # ic(encoding)
    encoding = adapt_e(encoding)
    # ic(encoding)

    # accompany chords
    note_chords = []
    for chord, note in zip(chords, encoding):
        cur_idx = note[0] * 2
        if note[1] >= pos_resolution * 2:
            cur_idx += 1
        if len(note_chords) < cur_idx:
            note_chords = note_chords + [NO_CHORD] * (cur_idx - len(note_chords))
        elif len(note_chords) == cur_idx:
            note_chords.append(chord)
        elif len(note_chords) == cur_idx + 1 and note_chords[-1] == NO_CHORD:
            note_chords[-1] = chord

    for i in range(1, len(note_chords)):
        if note_chords[i] == NO_CHORD:
            note_chords[i] = note_chords[i - 1]

    # create midi object
    midi_obj = encoding_to_midi(encoding)
    midi_obj.tempo_changes[0].tempo = TEMPO
    midi_obj.instruments[0].notes.sort(key=lambda x: (x.start, -x.end))
    midi_obj.instruments[0].name = 'melody'
    midi_obj.instruments.append(
        miditoolkit.Instrument(program=midi_obj.instruments[0].program, is_drum=False, name='chord')
        )
    midi_obj.instruments[0].program = 40
    midi_obj.instruments[1].notes = []
    ticks = midi_obj.ticks_per_beat

    lyrics = []
    for word in _lyrics:
        if word not in [',', '.', '?']:
            lyrics.append(word)
        else:
            lyrics[-1] += word

    ic(len(lyrics), len(midi_obj.instruments[0].notes))
    # ic(midi_obj.instruments)
    assert len(lyrics) == len(midi_obj.instruments[0].notes)

    for word, note in zip(lyrics, midi_obj.instruments[0].notes):
        midi_obj.lyrics.append(miditoolkit.Lyric(text=word, time=note.start))

    for idx, chord in enumerate(note_chords):
        if chord != NO_CHORD:
            root, _type = chord.split(':')
            root = pitch_dict[root]
            midi_obj.instruments[1].notes.append(
                miditoolkit.Note(
                    velocity=80, pitch=C2 + root, start=(idx * 2) * ticks, end=(idx * 2 + 2) * ticks
                    )
                )
            for shift in _CHORD_KIND_PITCHES[_type]:
                midi_obj.instruments[1].notes.append(
                    miditoolkit.Note(
                        velocity=80, pitch=C3 + (root + shift) % 12, start=(idx * 2) * ticks, end=(idx * 2 + 2) * ticks
                        )
                    )

    midi_obj.dump(midi_filename, charset='utf-8')
