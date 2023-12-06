import miditoolkit

def midi_to_lyrics(midi) -> dict:
    # Load MIDI file
    # midi = miditoolkit.MidiFile(midi_path, charset='utf-8')

    ticks_per_beat = midi.ticks_per_beat
    tempo = midi.tempo_changes[0].tempo

    def ticks_to_time(ticks):
        return ticks / ticks_per_beat * 60 / tempo

    # Extract note events and their timing information
    note_events = []
    for note in midi.instruments[0].notes:
        note_events.append({
            'start': ticks_to_time(note.start),
            'end': ticks_to_time(note.end),
            'pitch': pitch_to_note[note.pitch],
            'velocity': note.velocity
        })

    # Sort note events by start time
    note_events = sorted(note_events, key=lambda x: x['start'])

    # get lyric events with start time
    lyric_events = []
    for lyric in midi.lyrics:
        lyric_events.append({
            'start': ticks_to_time(lyric.time),
            'text': lyric.text
        })
    
    # Sort lyrics by start time
    lyric_events = sorted(lyric_events, key=lambda x: x['start'])

    # Initialize output variables
    notes = []
    notes_duration = []
    lyrics = []
    processed = False

    if note_events[0]['start'] > 0:
        lyrics.append('AP')
        notes.append(['rest'])
        notes_duration.append([note_events[0]['start']])

    def remove_nota(text):
        exist = False
        for notation in notations:
            if notation in text:
                text = text.replace(notation, '')
                exist = True

        return text, exist

    # Loop through note events and generate lyrics and note information
    while len(note_events) > 0 and len(lyric_events) > 0:
        note_event = note_events[0]

        # Convert note start and duration to time in seconds
        start_time = note_event['start']
        duration_time = note_event['end'] - note_event['start']

        if len(lyric_events) > 1 and start_time >= lyric_events[1]['start']:
            if bias > 0 and sep:
                notes_duration[-1][-1] -= bias
                lyrics.append('AP')
                notes.append(['rest'])
                notes_duration.append([bias])
                sep = False

            lyric_events.pop(0)
            processed = False

        if not processed:
            text, sep = remove_nota(lyric_events[0]['text'])
            lyrics.append(text)
            notes.append([])
            notes_duration.append([])
            processed = True

        notes[-1].append(note_event['pitch'])

        if len(note_events) > 1:
            duration_time = note_events[1]['start'] - note_event['start']
            bias = note_events[1]['start'] - note_event['end']
        else:
            duration_time = note_event['end'] - note_event['start']
            bias = 0
        
        notes_duration[-1].append(duration_time)

        if duration_time == 0:
            notes_duration[-1] = notes_duration[-1][:-1]
            notes[-1] = notes[-1][:-1]

        note_events.pop(0)

    i = 0
    while i < len(notes):
        if len(notes[i]) == 0:
            notes.pop(i)
            notes_duration.pop(i)
            lyrics.pop(i)
        else:
            i += 1

    assert len(lyrics) == len(notes) 
    assert len(lyrics) == len(notes_duration)
    # Return output dictionary
    output = {
        'text': ''.join(lyrics),
        'notes': ' | '.join([' '.join(note) for note in notes]),
        'notes_duration': ' | '.join([' '.join([str(d) for d in duration]) for duration in notes_duration]),
        'input_type': 'word'
    }
    print(sum(sum(notes_duration, [])))
    # print(output)
    return output


notations = [',', '.', '，', '。', '!', '?', '！', '？']
pitch_to_note = {
0: 'C-1',
1: 'C#-1/Db-1',
2: 'D-1',
3: 'D#-1/Eb-1',
4: 'E-1',
5: 'F-1',
6: 'F#-1/Gb-1',
7: 'G-1',
8: 'G#-1/Ab-1',
9: 'A-1',
10: 'A#-1/Bb-1',
11: 'B-1',
12: 'C0',
13: 'C#0/Db0',
14: 'D0',
15: 'D#0/Eb0',
16: 'E0',
17: 'F0',
18: 'F#0/Gb0',
19: 'G0',
20: 'G#0/Ab0',
21: 'A0',
22: 'A#0/Bb0',
23: 'B0',
24: 'C1',
25: 'C#1/Db1',
26: 'D1',
27: 'D#1/Eb1',
28: 'E1',
29: 'F1',
30: 'F#1/Gb1',
31: 'G1',
32: 'G#1/Ab1',
33: 'A1',
34: 'A#1/Bb1',
35: 'B1',
36: 'C2',
37: 'C#2/Db2',
38: 'D2',
39: 'D#2/Eb2',
40: 'E2',
41: 'F2',
42: 'F#2/Gb2',
43: 'G2',
44: 'G#2/Ab2',
45: 'A2',
46: 'A#2/Bb2',
47: 'B2',
48: 'C3',
49: 'C#3/Db3',
50: 'D3',
51: 'D#3/Eb3',
52: 'E3',
53: 'F3',
54: 'F#3/Gb3',
55: 'G3',
56: 'G#3/Ab3',
57: 'A3',
58: 'A#3/Bb3',
59: 'B3',
60: 'C4',
61: 'C#4/Db4',
62: 'D4',
63: 'D#4/Eb4',
64: 'E4',
65: 'F4',
66: 'F#4/Gb4',
67: 'G4',
68: 'G#4/Ab4',
69: 'A4',
70: 'A#4/Bb4',
71: 'B4',
72: 'C5',
73: 'C#5/Db5',
74: 'D5',
75: 'D#5/Eb5',
76: 'E5',
77: 'F5',
78: 'F#5/Gb5',
79: 'G5',
80: 'G#5/Ab5',
81: 'A5',
82: 'A#5/Bb5',
83: 'B5',
84: 'C6',
85: 'C#6/Db6',
86: 'D6',
87: 'D#6/Eb6',
88: 'E6',
89: 'F6',
90: 'F#6/Gb6',
91: 'G6',
92: 'G#6/Ab6',
93: 'A6',
94: 'A#6/Bb6',
95: 'B6',
96: 'C7',
97: 'C#7/Db7',
98: 'D7',
99: 'D#7/Eb7',
100: 'E7',
101: 'F7',
102: 'F#7/Gb7',
103: 'G7',
104: 'G#7/Ab7',
105: 'A7',
106: 'A#7/Bb7',
107: 'B7',
108: 'C8',
109: 'C#8/Db8',
110: 'D8',
111: 'D#8/Eb8',
112: 'E8',
113: 'F8',
114: 'F#8/Gb8',
115: 'G8',
116: 'G#8/Ab8',
117: 'A8',
118: 'A#8/Bb8',
119: 'B8',
120: 'C9',
121: 'C#9/Db9',
122: 'D9',
123: 'D#9/Eb9',
124: 'E9',
125: 'F9',
126: 'F#9/Gb9',
127: 'G9'
}
