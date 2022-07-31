import os
import string
from icecream import ic
from pypinyin import pinyin, Style
from tabulate import tabulate
from miditoolkit.midi import parser as mid_parser
import sys

from config import interval_range, WEAK_BEAT
from word_utils import get_in_word_pos
import score_config as sc
from tqdm import tqdm

LYRIC_PATH = "lyric.txt"

midi_suffix = ".mid"
src = sys.argv[1]
PUNC = [",", ".", "?"]
TICKS = 480

def get_pitch_score(song:sc.Song):
    def get_transition_score(delta_notes, prev_tone, curr_tone):
        note_range = interval_range[prev_tone][curr_tone]
        
        for i, note_r in enumerate(note_range):
            if note_r[0] <= delta_notes <= note_r[1]:
                return i
            
        return 3
    lyrics = song.lyric['text']
    sents_num = 0
    first_pitch_i = 0 
    count = [0, 0, 0, 0, 0] # [good, acceptable, not good, unacceptable, contour]
    for i, l in enumerate(lyrics):
        if i == 0:
            continue

        try:
            # Transition
            prev_tone, curr_tone = song.get_tone(i-1), song.get_tone(i)
            prev_pitch, curr_pitch = song.get_corres_notes(i-1).pitch, song.get_corres_notes(i).pitch
            
            t = get_transition_score(curr_pitch-prev_pitch, prev_tone, curr_tone)
            count[t] += 1
            
            # Contour
            if l[-1] in PUNC:
                if l[-1] in [',', '?']:
                    count[-1] += 1 if curr_pitch < song.get_pitch(first_pitch_i) else 0

                elif l[-1] in ['.']: 
                    count[-1] += 1 if curr_pitch > song.get_pitch(first_pitch_i) else 0
                
                sents_num += 1
                first_pitch_i = i

        except ValueError:
            continue
        except IndexError:
            continue
    
    count[-1] /=  sents_num
    score = sum([ w * c for w, c in zip(sc.pitch_weight, count)]) / len(lyrics)
    return count, 1-score

def get_rhythm_score(song:sc.Song):
    def get_aux_score(aux_flag, pos):
        if pos not in WEAK_BEAT and aux_flag:
            return 1

        return 0
    
    aux_count = 0
    lyrics = song.lyric['text']
    for i, (l, aux_flag) in enumerate(zip(lyrics, song.aux_mask)):
        time = song.lyric['time'][i]
        note_idx = song.get_noteidx_by_time(time)
        
        pos = song.get_pos(note_idx)
        aux_count += get_aux_score(aux_flag, pos)
        
        # if aux_flag:
            # ic(l, pos)
    
    aux_count /= sum([1 for a in song.aux_mask if a])
    count = [1-aux_count]
    score = sum([ w * c for w, c in zip(sc.rhythm_weight, count)])
    return count, score

def get_structure_score(song:sc.Song):
    sents_cnt = 1
    count = [0, 0]   # [missed rests, broken phrases]
    
    
    lyrics = song.lyric['text']
    word_pos = song.in_word_pos    
    for i, (l, w) in enumerate(zip(lyrics, word_pos)):
        try:
            prev_note = song.get_corres_notes(i-1)
            curr_note = song.get_corres_notes(i)
            next_note = song.get_corres_notes(i+1)
            
            prev_ed = prev_note.end
            curr_st = curr_note.start
            prev_interval = (curr_st - prev_ed) / TICKS
            curr_interval = (next_note.start - curr_note.end) / TICKS
            
            # missed rests
            if l[-1] in PUNC:
                sents_cnt += 1
                count[0] += 1 if curr_interval == 0 else 0

            # broken phrases
            if w and prev_interval > 0.5: 
                count[-1] += 1

        except IndexError:
            continue
    
    count[0] /= sents_cnt
    count[1] /= sum([1 for w in word_pos if w])
    
    count[0] = 1 - count[0]
    count[1] = 1 - count[1]
    score = sum([ w * c for w, c in zip(sc.struct_weight, count)])
    return count, score

def score(midi, midi_file_name, results):
    song = sc.Song(midi)
    song.print_debug()

    # Pitch score
    pitch_count, pitch_score = get_pitch_score(song)
    
    # Rhythm score
    rhythm_count, rhythm_score = get_rhythm_score(song)

    # Structure score
    structure_count, structure_score = get_structure_score(song)
    
    score = [ pitch_score, rhythm_score, structure_score ]
    score = sum([ w * s for w, s in zip(sc.weights, score)])

    results.append((midi_file_name, 
                    len(song.lyric['text']), score,
                    pitch_count, pitch_score, 
                    rhythm_count, rhythm_score,
                    structure_count, structure_score))
    return score

def print_info(modes_results):
    for mode_results in modes_results:
        total_score = 0
        total_pit_scr = 0 
        total_pit_cnt = [0, 0, 0, 0, 0]
        total_rhy_scr = 0
        total_rhy_cnt = [0, 0]
        total_stc_scr = 0
        total_stc_cnt = [0, 0]

        for row in mode_results:
            total_score += row[2]
            total_pit_cnt = [ a + b for a, b in zip(row[3], total_pit_cnt) ]
            total_pit_scr += row[4]
            total_rhy_cnt = [ a + b for a, b in zip(row[5], total_rhy_cnt) ]
            total_rhy_scr += row[6]
            total_stc_cnt = [ a + b for a, b in zip(row[7], total_stc_cnt) ]
            total_stc_scr += row[8]

        demo_cnt = len(mode_results)
        print("Number of Demo: ", demo_cnt)
        print("Average Final Score: {:.4}".format(total_score/demo_cnt))
        print("Average Pitch Score: {:.4}".format(total_pit_scr/demo_cnt))
        print("Average Pitch Count: {}".format( [p / demo_cnt for p in total_pit_cnt] ))
        print("Average Rhythm Score: {:.4}".format(total_rhy_scr/demo_cnt))
        print("Average Rhythm Count: {}".format([r / demo_cnt for r in total_rhy_cnt]))
        print("Average Structure Score: {:.4}".format(total_stc_scr/demo_cnt))
        print("Average Structure Count: {}".format([s / demo_cnt for s in total_stc_cnt]))
        
        headers = [
                    "Demo", "Sents length", "Score",
                    "Pitch Count", "Pitch Score",
                    "Rhythm Count", "Rhythm Score",
                    "Structure Count", "Structure Score",
                  ]
        print(tabulate(mode_results, headers=headers, tablefmt='fancy_grid'))
        print()

def split_results(results):
    """ Split results by their constraint modes"""
    modes_results = [[], []]
    for r in results:
        if str(config.ConstraintMode.N) in r[0]:
            modes_results[0].append(r)
        if str(config.ConstraintMode.PB_2) in r[0]:
            modes_results[1].append(r)
    
    modes_results = [ r for r in modes_results if r]
    return modes_results

def main():
    files = [ f"{root}/{f}" for root, _, files in os.walk(src) for f in files if f not in [".DS_Store", "score.txt"] and '.backup' not in root and 'a' not in root ]
    
    # debug 
    # file = ["倒带_BASE_ch_4566.mid", "达尔文_BASE_ch_4566.mid"]
    # files = [f"{src}/{f}" for f in file]
    
    files.sort()
    midis = [ mid_parser.MidiFile(f) for f in files ]
    print(files)
    results = []
    for idx, (m, f) in tqdm(enumerate(zip(midis, files))):
        print(f"Now processing {f}")
        score(m, f, results)

    # print([results])
    print_info([results])


if __name__ == "__main__":
    main()


