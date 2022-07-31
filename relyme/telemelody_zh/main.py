"""Main Pipelined for L2M with constraints
"""
from os import path, mkdir
from icecream import ic
from tqdm import tqdm
import infer
from config import GEN_MODE
from word_utils import clean, clean_beat, get_sents_pos, get_in_word_pos, get_sents_form, get_structure

DATA_DIR = "./data/"
CHORD_FILE = DATA_DIR + "chord.txt"
LYRIC_FILE = DATA_DIR + "lyric.txt"
STRCT_FILE = DATA_DIR + "structure.txt"

OUTPUT_DIR = "results"
BASE_OUTPUT = f"{OUTPUT_DIR}/base"
ReLyMe_OUTPUT = f"{OUTPUT_DIR}/relyme"

if not path.exists(OUTPUT_DIR):
    mkdir(OUTPUT_DIR)
    mkdir(BASE_OUTPUT)
    mkdir(ReLyMe_OUTPUT)

def infer_pipe(lyrics, _chords, midi_filename, struct=None):
    """Pipeline Generating L2M utilizing infer module

    Args:
        lyrics (str): lyrics
        _chords (list): chord progression
        midi_filename (str): output midi filename
        c_mode (ConstraintMode): specifying constraint modes
        struct (list): the structure of input lyrics
    """
    ic(midi_filename, lyrics, len(lyrics))

    # stage 1 L2B
    beat = infer.stage_1_get_l2b(list(lyrics.replace(',', '.')))

    # Perform Matched Strong/Weak positions constraints
    if GEN_MODE == "ReLyMe":
        beat = clean_beat(lyrics, beat)

    # stage 2 B2T
    trend, chords = infer.stage_2_get_b2t(lyrics, beat, _chords)

    flag = True
    if GEN_MODE == "ReLyMe":
        _lyrics = {
            "lyrics": get_sents_pos(lyrics),
            "struct": get_structure(lyrics, struct)
        }
        sent_info = {
            "word_pos": get_in_word_pos(lyrics),
            "sent_form": get_sents_form(lyrics)
        }
    else:
        _lyrics, sent_info = None, None

    while flag:
        try:
            # stage 3 T2M
            melody = infer.stage_3_get_t2m(trend, _lyrics, sent_info)

            # stage 4 Melody2Midi
            infer.stage_4_get_m2m(melody, chords, lyrics, midi_filename)

        # keep generating until len(note) == len(lyrics)
        except AssertionError:
            print("Assertion Error, generate again")
            continue
        flag = False

    print(f"{midi_filename} Done!\n\n")

def main():
    lyrics = [] # lyrics
    strcts = [] # structure of lyrics
    chords = [] # chords

    with open(CHORD_FILE, 'r') as c:
        tmp = c.readlines()
        chords = [ chord.split() for chord in tmp ]

    with open(LYRIC_FILE, 'r') as l:
        lyrics = l.readlines()
        lyrics = [ clean(l) for l in lyrics ]

    with open(STRCT_FILE, 'r') as s:
        tmp = s.readlines()
        strcts = [ strct.split() for strct in tmp ]

    ic(lyrics, chords, strcts)    

    for idx, (ly, ch, st) in tqdm(enumerate(zip(lyrics, chords, strcts))):
        output_dir = BASE_OUTPUT if GEN_MODE == "BASE" else ReLyMe_OUTPUT
        output_path = path.join(output_dir, f"{idx}_{GEN_MODE}.mid")

        if GEN_MODE == "BASE":
            infer_pipe(ly, ch, output_path)

        elif GEN_MODE == "ReLyMe":
            infer_pipe(ly, ch, output_path, st)


if __name__ == "__main__":
    main()