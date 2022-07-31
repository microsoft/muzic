import os
import sys
import re
import shutil
from icecream import ic
sys.path.append(os.path.join(os.getcwd(), "score"))

import score.score_en as score_en


if __name__ == "__main__":
    midi_dir   = sys.argv[1]
    output_dir = sys.argv[2]
    midi_reg = re.compile('(.*mid$)')

    song_prefices = [
        os.path.join(root, file[:-4])
        for root, _, files in os.walk(midi_dir)
        for file in files
        if midi_reg.match(file)
    ]
    scores = [
        score_en.get_score(song)
        for song in song_prefices
    ]
    
    best_score = max(scores)
    best_index = scores.index(best_score)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(f"{song_prefices[best_index]}.mid", output_dir)
    
    ic(scores)