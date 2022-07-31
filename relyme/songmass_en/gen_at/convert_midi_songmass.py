import os
import sys
from miditoolkit.midi import parser as mid_parser, containers
from icecream import ic

src = sys.argv[1]
dst = sys.argv[2]
K = sys.argv[3] if len(sys.argv) > 3 else ''

align_file = K+"align_result.txt"
midi_file  = K+"gen_.mid"
output_suffix = "_songmass.mid"

rest_note_filter = False # 如果生成的结果太多休止符, 则舍去此样本

if not os.path.exists(dst):
    os.makedirs(dst)

def convert(demo_i):
    print(demo_i)
    with open(f"{src}/{demo_i}/{align_file}", "r") as f:
        lines = f.readlines()

    lyrics_ali = []
    note_cnt = 0
    for i in range(int(len(lines) / 2)):
        lyrics = lines[i * 2 + 1].strip('\n').split(' ')
        
        k = len(lyrics) - 1
        while k > 0:
            if lyrics[k] != "":
               lyrics[k] += '.'
               break 
            k -= 1
        
        # ic(lyrics)
        
        cnt = 0
        for c in lyrics:
            if c == '':
                cnt += 1
                continue
            
            if c != '' and cnt > 0:
                blank = ['N'] * int(cnt/2)
                lyrics_ali += blank
                cnt = 0
            
            lyrics_ali.append(c)

        note_cnt += len(lines[i * 2].strip('\n').split(' '))
        
        if len(lyrics_ali) < note_cnt:
            lyrics_ali += ['N'] * (note_cnt - len(lyrics_ali))

    if note_cnt < len(lyrics_ali):
        for i in range(len(lyrics_ali) - note_cnt):
            lyrics_ali.remove('N')

    assert note_cnt == len(lyrics_ali) 

    a = lines[0].split()

    if a.count('-') / len(a) > 0.25 and rest_note_filter:
        """
        生成结果太多休止符 (-), 则放弃生成此样例
        """
        print(f"Too many rest notes in {demo_i}")
        return

    midi = mid_parser.MidiFile(f"{src}/{demo_i}/{midi_file}")
    melody = midi.instruments[0].notes
    
    print(melody[-1])
    print((lyrics_ali))
    if len(melody) < len(lyrics_ali):      
        for i in range(len(lyrics_ali) - len(melody)):
            last_note = melody[-1]
            stuff = containers.Note(last_note.velocity, last_note.pitch, last_note.end, last_note.end + last_note.end - last_note.start)
        
            melody.append(stuff)
            print(melody[-1])
    print(len(melody))
    print(lyrics_ali)
    for i, p in enumerate(lyrics_ali):
        if lyrics_ali[i] != 'N':
            word = containers.Lyric(lyrics_ali[i], melody[i].start)
            midi.lyrics.append(word)

    print(midi.lyrics)
    midi.dump(f"{dst}/{K}{demo_i}{output_suffix}", charset='utf-8')


if __name__ == "__main__":
    for d in os.listdir(src):
    # for d in range(3, 4):
        if d in ['.DS_Store']: continue
        convert(str(d))
