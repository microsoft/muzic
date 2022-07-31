from os import makedirs, path
from icecream import ic

from md import get_md
from pd_dd import get_pd_dd
from config import interval_range, PUNC, WEAK_BEAT, TICKS
from midi_adjust import split_midi, dump_midi
from SongWrapper import Song
import score_config as sconfig

def get_pitch_score(song:Song):
    def get_transition_score(delta_note, prev_tone, curr_tone):
        note_range = interval_range[prev_tone][curr_tone]
        
        for i, note_r in enumerate(note_range):
            if note_r[0] <= delta_note <= note_r[1]:
                return i
            
        return 3
    
    lyrics = song.lyric['text']
    first_pitch = song.get_pitch(0)
    contour = 0
    transition = [0, 0, 0, 0] # [good, acceptable, not good, unacceptable]
    for i, l in enumerate(lyrics):
        if i == 0: continue

        try:
            # Transition
            prev_tone, curr_tone = song.get_tone(i-1), song.get_tone(i)
            prev_pitch, curr_pitch = song.get_last_note(i-1).pitch, song.get_first_note(i).pitch
            
            t = get_transition_score(curr_pitch-prev_pitch, prev_tone, curr_tone)
            transition[t] += 1
            
            # Contour
            if l[-1] in PUNC:
                if l[-1] in ['.'] and curr_pitch < first_pitch or \
                   l[-1] in [',', '?'] and curr_pitch > first_pitch:
                    contour += 1
                
                first_pitch = song.get_first_note(i+1).pitch

        except ValueError:
            continue
        except IndexError:
            continue
    
    contour_score = contour / song.n_sents
    transition_score = sum([ w * c for w, c in zip(sconfig.transition_weight, transition)]) / song.n_notes
    
    pitch_detail = [transition_score, contour_score]
    pitch_score  = sum([ w * c for w, c in zip(sconfig.pitch_weight, pitch_detail)])
    return pitch_detail, pitch_score

def get_rhythm_score(song:Song):
    # strong weak positions score
    def get_sw_score(song:Song):
        def aux_rule(aux_flag, pos):
            # return 0 if aux word on strong beat
            # else 1
            if pos not in WEAK_BEAT and aux_flag:
                return 0
            return 1
        
        def key_rule(key_flag, pos):
            # return 0 if keyword on weak beat
            # else 1
            if pos in WEAK_BEAT and key_flag > -1:
                return 0
            return 1
        
        aux_count = 0
        key_count = 0
        for pos, aux_flag, key_flag in zip(song.pos, song.aux_mask, song.key_mask):
            aux_count += aux_rule(aux_flag, pos)
            key_count += key_rule(key_flag, pos)
        
        aux_count /= song.n_notes
        key_count /= song.n_notes
        
        sw_detail = [aux_count, key_count]
        sw_score  = sum([ w * c for w, c in zip(sconfig.sw_weight, sw_detail)])
        return sw_detail, sw_score
        
    # paused positions score
    def get_ps_score(song:Song):
        # missed rests
        mr_count = 0   
        lyrics = song.lyric['text']
        for i, l in enumerate(lyrics):
            try:
                next_note = song.get_first_note(i+1)
                curr_last_note  = song.get_last_note(i)
                
            except IndexError:
                mr_count += 1
                break
            
            curr_ed = curr_last_note.end
            curr_interval = (next_note.start - curr_ed) / TICKS
            
            # missed rests
            if l[-1] in PUNC:
                mr_count += 1 if curr_interval != 0 else 0
                                
        # broken phrases
        bp_count = 0   
        word_pos = song.in_word_pos    
        for i, w in enumerate(word_pos):
            try:
                prev_note = song.get_note(i-1)
                curr_note = song.get_note(i)
                
            except IndexError:
                continue
            
            prev_ed = prev_note.end
            curr_st = curr_note.start
            prev_interval = (curr_st - prev_ed) / TICKS

            if w and prev_interval < 0.5: 
                bp_count += 1
            
        mr_count /= song.n_sents
        bp_count /= sum([1 for w in word_pos if w])
        
        ps_detail = [mr_count, bp_count]
        ps_score  = sum([ w * c for w, c in zip(sconfig.ps_weight, ps_detail)])
        return ps_detail, ps_score

    sw_detail, sw_score = get_sw_score(song)
    ps_detail, ps_score = get_ps_score(song)
    # ic(sw_detail, ps_detail)
    rhythm_detail = [sw_score, ps_score]
    rhythm_score  = sum([ w * c for w, c in zip(sconfig.rhythm_weight, rhythm_detail)])
    
    return rhythm_detail, rhythm_score
    
def get_structure_score(song:Song):
    # split the original song to two counterpart based on the structure info
    STRCT_TMP = "./strct_temp"
    strct_a = f"{STRCT_TMP}/a"
    strct_b = f"{STRCT_TMP}/b"
    
    makedirs(STRCT_TMP, exist_ok=True)
    makedirs(strct_a, exist_ok=True)
    makedirs(strct_b, exist_ok=True)
    
    midi_n = f"{song.song_name}.mid"
    if not path.exists(f"{strct_a}/{midi_n}"):
        obj_1, obj_2 = split_midi(song.midi, song.strct)
        dump_midi(f"{strct_a}/{midi_n}", obj_1)
        dump_midi(f"{strct_b}/{midi_n}", obj_2)
    
    # pitch distribution, duration distribution
    pd_score, dd_score = get_pd_dd(f"{strct_a}/{midi_n}", 
                                   f"{strct_b}/{midi_n}")
    
    # melody distance
    md_score = get_md(f"{strct_a}/{midi_n}", 
                      f"{strct_b}/{midi_n}")
    
    pd_score = pd_score[0]
    dd_score = dd_score[0]
    md_score = md_score[0]
    
    struct_detail = [pd_score, dd_score, md_score]
    struct_score = sum([ w * c for w, c in zip(sconfig.struct_weight, struct_detail)])
                    
    return struct_detail, struct_score

def get_score(song_prefix):
    """get the overall objective score and detailed score of each attribute

    Args:
        song_prefix: prefix for the song, which should have 
        - {song_prefix}.mid (midi files of the song)
        - {song_prefix}.strct (structure info of the song)
    """
    
    song = Song(song_prefix)
    song.print_debug()

    # pitch score
    pitch_detailed, pitch_score = get_pitch_score(song)
    ic(pitch_detailed, pitch_score)
    print()
    
    # rhythm score
    rhythm_detail, rhythm_score = get_rhythm_score(song)
    ic(rhythm_detail, rhythm_score)
    print()
    
    # structure score
    struct_detail, struct_score = get_structure_score(song)
    ic(struct_detail, struct_score)
    print()
    
    score_detail = [pitch_score, rhythm_score, struct_score]
    score = sum([ w * c for w, c in zip(sconfig.overall_weights, score_detail)])
    
    return score
    
if __name__ == "__main__":
    score = get_score("testmid/zh/tele-zh")
    ic(score)