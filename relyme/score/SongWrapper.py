from pypinyin import lazy_pinyin, Style
from miditoolkit.midi import parser as mid_parser
from icecream import ic

from config import TICKS, PUNC


class Song():
    def __init__(self, song_prefix, lang="zh") -> None:
        self.song_prefix = song_prefix
        self.song_name   = song_prefix.split('/')[-1]
        self.lang = lang
        self.midi = mid_parser.MidiFile(f"{song_prefix}.mid")
        self.lyric   = self.lyr_transform(self.midi.lyrics)
        self.n_sents = max(1, self.get_sents_num()) # number of sentences
        self.melody  = self.mld_transform(self.midi.instruments[0].notes)
        self.n_notes = len(self.melody) # number of notes
        self.pos   = self.pos_transform()
        self.strct = self.strct_transform(f"{song_prefix}.strct")
        
        if self.lang == "zh":
            from word_utils_zh import get_aux_mask, get_keyword_mask, get_in_word_pos
            self.tone = self.tone_transform()
            lyric_string = ''.join(self.lyric['text'])
            self.aux_mask = get_aux_mask(lyric_string)
            self.key_mask = get_keyword_mask(lyric_string)
            self.in_word_pos = get_in_word_pos(lyric_string)

        elif self.lang == "en":
            from word_utils_en import get_aux_mask, get_keyword_mask, get_in_word_pos
            self.syl_sents = self.syl_transform(f"{song_prefix}.syl")
            lyric_string = ' '.join(self.lyric['text'])
            self.aux_mask = get_aux_mask(lyric_string, self.syl_sents)
            self.key_mask = get_keyword_mask(lyric_string, self.syl_sents)
            self.in_word_pos = get_in_word_pos(self.syl_sents)
            
    def lyr_transform(self, _lyrics):
        lyric = {
            "text": [ l.text for l in _lyrics ],
            "time": [ l.time for l in _lyrics ]
        }

        return lyric 
    
    def syl_transform(self, syllable_file):
        with open(syllable_file, "r") as s:
            syl = s.readlines()
        
        return syl[0]
    
    def get_sents_num(self):
        return sum([
            1
            for l in self.lyric["text"]
            if l[-1] in PUNC
        ])

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
    
    def strct_transform(self, strct_file):
        with open(strct_file, "r") as s:
            strct = s.readlines()
        
        return strct[0].split()
    
    def pos_transform(self):
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
    
    def get_last_note(self, lyr_idx):
        # get the last note of input lyric index
        # (used when a word corresponds to multiple notes)
        try:
            last_note_id = self.get_noteidx_by_time(self.lyric["time"][lyr_idx+1]) - 1
        except IndexError:
            last_note_id = self.n_notes-1

        return self.get_note(last_note_id)
    
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
            if note.start == start: return i
        
        return 0
    
    def get_first_note(self, lyr_idx):
        """Get the first note of the lyr_idx
        """
        time = self.lyric['time'][lyr_idx]
        note_idx = self.get_noteidx_by_time(time)
        
        return self.get_note(note_idx)
    
    def get_word_pos(self, lyr_idx):
        return self.in_word_pos[lyr_idx]
    
    def print_debug(self):
        ic(self.lang)
        ic(self.song_name)
        ic(self.song_prefix)
        print(self.lyric)
        ic(self.n_sents)
        ic(self.strct)
        print(self.aux_mask)
        print(self.key_mask)
        print(self.in_word_pos)
        if self.lang == "en":
            ic(self.syl_sents)