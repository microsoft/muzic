from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi.containers import TempoChange, Note, Lyric, Instrument
from icecream import ic

INPUT = "zz_final_mid_zh/no-rhythm"
LYRICS_DIR = "L2M_TEST"
OUTPUT = "zz_final_mid_zh/no-shythm"
DEFAULT_INST = 0 # Default instruments aka melody

def load_midi(midi_file) -> mid_parser.MidiFile:
    """Load midi
    if it's already loaded, return directly
    else load the mid
    """
    if isinstance(midi_file, mid_parser.MidiFile):
        return midi_file
    
    return mid_parser.MidiFile(midi_file)

def dump_midi(midi_file, midi_obj):
    midi_obj.dump(midi_file, charset='utf-8')

def get_notes(midi_obj, instrument_idx=DEFAULT_INST):
    midi_obj = load_midi(midi_obj)

    return midi_obj.instruments[instrument_idx].notes

def get_atrributes(midi_obj, attr):
    """get attributes of the midi_obj

    Args:
        midi_obj (_type_):
        attr (list): e.g. [(tempo, None), (instruments, 0), (lyrics, None)]

    Returns:
        dict: e.g. {"tempo": midi_obj.tempo_changes}
    """
    midi_obj = load_midi(midi_obj)


    results = {}
    for key in attr:
        if key[0] == "tempo":
            results["tempo"] = midi_obj.tempo_changes

        if key[0] == "instruments":
            results[f"instruments-{key[1]}"] = midi_obj.instruments[key[1]]

        if key[0] == "lyrics":
            results["lyrics"] = midi_obj.lyrics

    return results

def shift_notes_time(notes, offset):
    shift_notes = notes.copy()
    
    for n in shift_notes:
        n.start += offset
        n.end += offset
    
    return shift_notes

def shift_lyrics_time(lyrics, offset):
    shift_lyrics = lyrics.copy()
    
    for l in shift_lyrics:
        l.time += offset
    
    return shift_lyrics


def get_noteidx_by_time(melody, start):
    for i, note in enumerate(melody):
        if note.start == start:
            return i
    
    return 0

def split_midi(midi_file, strct):
    midi_obj = load_midi(midi_file)
    melody = midi_obj.instruments[0].notes
    lyrics = midi_obj.lyrics
    # ic(lyrics)
    
    # split melody to phrases based on punctuation
    lyr_curr_st = 0
    mld_curr_st = 0
    mld_next_st = -1
    tmp_mld = []
    tmp_lyr = []
    for i, ly in enumerate(lyrics):
        if ly.text[-1] in ['.', ',']:
            try:
                mld_next_st = get_noteidx_by_time(melody, lyrics[i+1].time)
            except IndexError:
                mld_next_st = len(melody)
            
            tmp_mld.append(melody[mld_curr_st:mld_next_st])
            tmp_lyr.append(lyrics[lyr_curr_st:i+1])
            lyr_curr_st = i + 1
            mld_curr_st = mld_next_st

    # ic(tmp_lyr)
    # ic(tmp_mld)
    
    # find the split index
    tmp_1 = []
    tmp_2 = []
    split_index = -1
    for i, s in enumerate(strct):
        if s not in tmp_1:
            tmp_1.append(s)
        
        else:
            split_index = i
            break
    tmp_2 = strct[split_index:]

    # ic(tmp_1, tmp_2)
    # ic(tmp_mld)

    midi_obj_1 = mid_parser.MidiFile()
    midi_obj_2 = mid_parser.MidiFile()

    midi_obj_1.instruments.append(Instrument(program=0, is_drum=False, name="melody"))
    midi_obj_2.instruments.append(Instrument(program=0, is_drum=False, name="melody"))
    
    midi_obj_1.lyrics = []
    midi_obj_2.lyrics = []
    midi_obj_1.temp_changes = midi_obj.tempo_changes
    midi_obj_2.temp_changes = midi_obj.tempo_changes

    first_ticks = -1
    for i_1, t_1 in enumerate(tmp_1):
        if t_1 in tmp_2:
            i_2 = tmp_2.index(t_1)

            if first_ticks == -1:
                first_ticks = -tmp_mld[len(tmp_1) + i_2][0].start + tmp_mld[i_1][0].start
            
            
            midi_obj_1.instruments[0].notes += tmp_mld[i_1]
            midi_obj_2.instruments[0].notes += shift_notes_time(tmp_mld[len(tmp_1) + i_2], first_ticks)

            midi_obj_1.lyrics += tmp_lyr[i_1]
            midi_obj_2.lyrics += shift_lyrics_time(tmp_lyr[len(tmp_1) + i_2], first_ticks)
        
    return midi_obj_1, midi_obj_2