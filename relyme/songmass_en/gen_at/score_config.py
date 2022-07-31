# Final score
# [Pitch_weight, Rhythm_weight, Structure_weight]
weights = [0.4, 0.3, 0.4]

# Pitch
# [Good, Acceptable, Not good, Unacceptable, Out of key]
pitch_weight = [0, 0.2, 0.5, 1, 1]

# Rhythm
# [Keyword, Auxiliary word]
rhythm_weight = [1]

# Structure
struct_weight = [0.5, 0.5]

from pypinyin import lazy_pinyin, Style
from word_utils import get_aux_mask, get_in_word_pos
from miditoolkit.midi import parser as mid_parser
from icecream import ic
TICKS = 480

class Song():
    def __init__(self, midi:mid_parser.MidiFile) -> None:
        self.lyric = self.lyr_transform(midi.lyrics)
        self.tone = self.tone_transform()
        self.melody = self.mld_transform(midi.instruments[0].notes)
        self.pos = self.get_all_pos()
        self.aux_mask = get_aux_mask(''.join(self.lyric['text']))
        self.in_word_pos = get_in_word_pos(''.join(self.lyric['text']))
        
        # assert len(self.lyric) == len(self.tone) == len(self.aux_mask)

    def lyr_transform(self, _lyrics):
        lyric = {
            "text": [],
            "time": []
        }
        
        lyric['text'] = [ l.text for l in _lyrics ]
        lyric['time'] = [ l.time for l in _lyrics ]
        
        return lyric 
    
    def mld_transform(self, _melody):
        return _melody
    
    def tone_transform(self):
        def get_tone_id(s) -> int:
            if s in ["1", "2", "3", "4"]:
                return int(s) - 1
            return 4
        
        lyric = [ l.replace('.', '') for l in self.lyric['text'] ]
        tone = lazy_pinyin(lyric, style=Style.TONE3)
        tone = [ get_tone_id(t[-1]) for t in tone ]
    
        return tone
        
    def get_all_pos(self):
        pos = [
            (note.start // TICKS) % 4
            for note in self.melody
        ]
            
        return pos
            
    def get_pos(self, note_idx):
        note_idx = min(note_idx, len(self.pos)-1)
        return self.pos[note_idx]
    
    def get_note(self, note_idx):
        note_idx = min(note_idx, len(self.melody)-1)
        return self.melody[note_idx]
    
    def get_pitch(self, note_idx):
        return self.get_note(note_idx).pitch
    
    def get_lyric(self, lyr_idx):
        lyr_idx = min(lyr_idx, len(self.lyric)-1)
        return self.lyric['text'][lyr_idx]
    
    def get_tone(self, lyr_idx):
        lyr_idx = min(lyr_idx, len(self.lyric)-1)
        return self.tone[lyr_idx]
    
    def get_aux_flag(self, lyr_idx):
        lyr_idx = min(lyr_idx, len(self.lyric)-1)
        return self.aux_mask[lyr_idx]
    
    def get_noteidx_by_time(self, start):
        for i, note in enumerate(self.melody):
            if note.start == start:
                return i
        
        return 0
    
    def get_corres_notes(self, lyr_idx):
        """Getting corresponding notes of the lyr_idx
        """
        time = self.lyric['time'][lyr_idx]
        note_idx = self.get_noteidx_by_time(time)
        
        return self.get_note(note_idx)
    
    def get_word_pos(self, lyr_idx):
        return self.in_word_pos[lyr_idx]
    
    def print_debug(self):
        print(self.lyric)
        # ic(self.melody)
        # ic(self.pos)
        # ic(self.tone)
        print(self.aux_mask)
        