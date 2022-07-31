# coding: utf-8

"""
Why: 
    把 songmass infer 的 notes 生成结果转换成 midi (无对齐信息、无歌词)

What:
    input:
    seq = "62 131 62 132 62 129 64 130 62 134"
        
    output:
    xxx.mid
"""

import os
import sys
import miditoolkit

get_last = True
find_structure = True
generated_file = False

num = sys.argv[1]

data_dir = "l2m_merge_" + num

output_dir = 'l2m_merge_' + num
output_suffix = '_.mid'
song_id_file = f'test_data/para/song_id.txt'

i = 0
beam = 5
SEP = '[sep]'
ALIGN = '[align]'

BPM  = 75 # BPM
VELOCITY = 80
REST_NOTE = 128 # 没有音符/休止符的index


# k = sys.argv[1] if len(sys.argv) > 1 else ''
k = ''

# MELODY INDEX:
# 0-127 : Notes
# 128 : Rest Notes
# 129-136 : Duration [0.25; 0.5 ;1 ;2 ;4 ;8 ;16 ;32]...


du_voc = dict([(129+i,str(x/100))  for i,x in enumerate(list(range(25,3325,25)))])
# print(du_voc)


def get_pitch_duration_structure(note_seq):
    seq = []
    
    #遍历寻找pitch-duration的结构
    #当有不合法情况出现时，找最后一个pitch和第一个duration，保证其相邻
    #p1 d1 p2 p3 d2 p4 d3-> p1 d1 p3 d1 p4 d3
    #p1 d1 p2 d2 d3 p3 d4-> p1 d1 p2 d2 p3 d4
    #p1 d1 p2 p3 d2 d3 p4 d4 -> p1 d1 p3 d2 p4 d4
    
    i = 0
    while (i<len(note_seq)):
        if note_seq[i] > REST_NOTE:
            # Duration
            i += 1
        else:
            # Pitch
            if i+1>=len(note_seq):
                # No Duration Followed
                break
            if note_seq[i+1] <= REST_NOTE:
                # Followed by a pitch
                i += 1

            pitch = int(note_seq[i])
            duration = int(note_seq[i+1])
            
            seq.append((pitch, duration))
            i += 2
    return seq

def separate_sentences(x, convert_int = False, find_structure = False):
    lst = x.copy()
    sep_positions = [i for i,x in enumerate(lst) if x==SEP]
    sep_positions.insert(0,-1)

    ret = []
    for i in range(len(sep_positions)-1):
        sent = lst[sep_positions[i]+1:sep_positions[i+1]] #SZH: not include sep token
        if convert_int:
            sent = list(map(int, sent))
        if find_structure:
            sent = list(map(int, sent))
            sent = get_pitch_duration_structure(sent)
        ret.append(sent)
    return ret


def get_notes(melody_seq, align):
    align_seq = align.strip('\n').split()
    # print(align_seq)

    # 过滤 ALIGN
    melody_seq = [ int(m) if m != SEP else m for m in melody_seq]
    notes = []
    global i

    avail_note = i
    # print(melody_seq[avail_note])
    try:
        while melody_seq[avail_note] == SEP or melody_seq[avail_note] >= REST_NOTE:
            avail_note += 2
    except Exception:
        print(avail_note, i, len(melody_seq))
        avail_note -= 2
        while melody_seq[avail_note] == SEP or melody_seq[avail_note] >= REST_NOTE:
            avail_note -= 1
        print(melody_seq[avail_note:])
        # assert False

    for a in align_seq:
        if a == '-':
            # print(f"i: {i}")
            # print(f"av: {avail_note}")
            # print(melody_seq[avail_note])

            assert melody_seq[avail_note] < 128
            notes.append(melody_seq[avail_note])
            notes.append(melody_seq[avail_note+1])

        else:
            notes.append(melody_seq[i])
            notes.append(melody_seq[i+1])
            if melody_seq[i] < REST_NOTE: avail_note = i
            i += 2
    
    while melody_seq[i] != SEP:
        notes.append(melody_seq[i])
        notes.append(melody_seq[i+1])
        i += 2
    i += 1    

    note_seq = get_pitch_duration_structure(notes)
    # print(note_seq)
    return note_seq

def gen_midi(midi_obj, note_seq):
    melody = miditoolkit.midi.containers.Instrument(program=1, is_drum=False, name="melody")
    tick = 0

    for note in note_seq:
        if note[0] < 128: # Pitch
            note_start = tick
            note_end   = tick + midi_obj.ticks_per_beat * float(du_voc[note[1]])
            n = miditoolkit.containers.Note(VELOCITY, note[0], int(note_start), int(note_end))

            melody.notes.append(n)
            tick = note_end
        else: # Rest
            assert note[0] == REST_NOTE
            tick += midi_obj.ticks_per_beat * float(du_voc[note[1]])
    
    midi_obj.instruments.append(melody)

    ts = miditoolkit.midi.containers.TimeSignature(4, 4, 0)
    tempo = miditoolkit.containers.TempoChange(tempo=BPM, time=0)
    midi_obj.tempo_changes.append(tempo)
    midi_obj.time_signature_changes.append(ts)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if __name__ == "__main__":
    for d in os.listdir(data_dir):
    # for d in range(3, 4):
        if d in ['.DS_Store']: continue
        i = 0
        print(d, "--------")
        with open(os.path.join(data_dir, f"{d}/melody"), "r") as m:
            seq = m.readlines()
            mel_seq = seq[0].split()
        with open(os.path.join(data_dir, f"{d}/{k}align_result.txt"), "r") as f:
            lines = f.readlines()

        note_seq = []
        #try:
        for idx in range(int(len(lines) / 2)):
            result = get_notes(mel_seq, lines[2 * idx])
            note_seq += result
    
        # except Exception:
        #     print(d, "not done")
        #     continue
        # print(d, len(note_seq))
        midi_obj = miditoolkit.midi.parser.MidiFile()
        gen_midi(midi_obj, note_seq)

        midi_obj.dump(os.path.join(output_dir, f"{d}/{k}gen{output_suffix}"))

# note_seq = get_notes(seq)
# gen_midi(note_seq)
