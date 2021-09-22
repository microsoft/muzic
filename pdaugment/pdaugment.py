#!/usr/bin/env python
# coding: utf-8

from ast import Index
import os
import math
import json
import sys
import numpy as np
import pickle
import pandas as pd
import librosa
import pyworld as pw
import soundfile as sf
import miditoolkit
import midiconvert as md
import random
import multiprocessing
import secrets
import string


# read metadata information of the libritts dataset
def read_meta_data(meta_data):
    """
    @params
    meta_data: using pd.read_csv to read csv file
    @return
    meta_datas: mapping list consists of wave_name, phone and new_phone.
    """
    res = []
    for index, row in meta_data.iterrows():
        path = row["wav"]
        wave_name = row["new_wav"]
        phone = row["phone"]
        new_phone = row["new_phone"]
        res.append((path,wave_name,phone, new_phone))

    return res

# convert pitch in frequency to midi number
def hz2midi(frequency):
    """
    @params
    frequency: pitch in frequency
    @return
    midi_number: pitch in midi number
    """
    return (69 + 12 * math.log((frequency/440), 2))

def midi2hz(midi):
    """
    @params
    midi: midi data
    @return
    tar_fre: target frequency
    """
    note, octave = md.number_to_note(midi)
    tar_fre = fre['{}'.format(octave)]['{}'.format(note)]
    return tar_fre

# determine the phone is vowel or not
def isVowel(phone):
    """
    @params
    phone: input phone
    @return
    boolean: the phone is vowel or not
    """
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    for vowel in vowels:
        if vowel in phone:
            return True
    
    return False

# convert midi to notes info
def midi2notes(midi_path):
    """
    @params
    midi_path: midi path of the item
    @return
    data: data is a tuple consists of the pitch, duration and interval information of the given midi file
    """
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    data = []
    notes = midi_obj.instruments[0].notes
    mapping = midi_obj.get_tick_to_time_mapping()
    for i in range(len(notes)):
        note = notes[i]
        st = mapping[note.start]
        end = mapping[note.end]
        if i != len(notes)-1:
            note1 = notes[i+1]
            next_st = mapping[note1.start]
        else:
            next_st = end
        data.append((note.pitch, end-st, next_st-end))
    return data

# extract syllables from wav
def get_syllables(wav_data, phone, new_phone):
    """
    @params
    wav_data: the mel attributes of the audio from pickle file
    phone: original phone from meta_data.csv
    new_phone: new phone from meta_data.csv
    @return
    syllables: syllable list consists of phoneme list and the start and end of the syllables
    """
    syllables = []
    
    result = []
    word_phones = phone.split(" / ")
    syllable_index = 0
    for word in word_phones:
        if "punc_" in word:
            continue

        temp_res = []

        phones = word.split(" ")
        for phone in phones:
            if phone == "-":
                result.append(temp_res)
                temp_res = []
            else:
                temp_res.append(syllable_index)
                syllable_index += 1

        if temp_res != []:
            result.append(temp_res)

    new_phone_components = new_phone.split(" ")
    phone_index = 0
    
    for syllable in result:
        if new_phone_components[phone_index] == "<BOS>":
            phone_index += 1
        elif new_phone_components[phone_index] == "sil":
            wav_start = wav_data[phone_index]
            wav_end = wav_data[phone_index + 1]
            
            syllables.append(([new_phone_components[phone_index]], [(wav_start, wav_end)]))
            
            phone_index += 1
        
        single_syllable_phoneme_list = new_phone_components[phone_index : (phone_index + len(syllable))]
        wav_mels = []
        for delta in range(0, len(syllable)):
            wav_start = wav_data[phone_index + delta]
            wav_end = wav_data[phone_index + delta + 1]
            wav_mels.append((wav_start, wav_end))
        
        syllables.append((single_syllable_phoneme_list, wav_mels))
        
        phone_index += len(syllable)
        pass
    
    return syllables

# Part 1: Determine the correspondence between notes and syllables (one-to-many or many-to-one) according to the duration of MIDI and speech.
def note_syllable_mapping(notes, syllables):
    """
    Calculate the correspondence between notes and syllables
    @params
    notes: note list consists of pitch, duration and interval.
    syllables: syllable list consists of phoneme list and the start and end of the syllables
    @return
    mappings: mapping list consists of note list, phoneme list, the start and end of the syllables and output rate of the wav.
    """
    
    INTERVAL = 12.5
    LOWWER_RATE = 0.5
    UPPER_RATE = 2
    
    mappings = []
    syllable_index = 0
    midi_note_index = 0
    
    while syllable_index < len(syllables):
        note = []
        
        all_phonemes = []
        all_wav_data = []
        
        phoneme_list, wav_data = syllables[syllable_index]
        
        all_phonemes += phoneme_list
        all_wav_data += wav_data
                
        wav_start = int(wav_data[0][0])
        wav_end = int(wav_data[len(wav_data) - 1][1])
        
        note.append(notes[midi_note_index])
        
        output_rate = 1

        curr_syllable_interval = wav_end*INTERVAL/1000 - wav_start*INTERVAL/1000
        curr_note_interval = notes[midi_note_index][1]
        
        syllable_flag = 0
        midi_note_flag = 0
        
        # mapping strategy
        while True:
            output_rate = curr_syllable_interval / curr_note_interval
            if output_rate < LOWWER_RATE:
                if midi_note_flag == 1:
                    output_rate = LOWWER_RATE
                    break
                syllable_index += 1
                syllable_flag = 1
                if syllable_index >= len(syllables):
                    output_rate = LOWWER_RATE
                    break
                phoneme_list, wav_data = syllables[syllable_index]
                
                all_phonemes += phoneme_list
                all_wav_data += wav_data
                
                wav_end = int(wav_data[len(wav_data) - 1][1])
                curr_syllable_interval = wav_end*INTERVAL/1000 - wav_start*INTERVAL/1000
            elif output_rate > UPPER_RATE:
                if syllable_flag == 1:
                    output_rate = UPPER_RATE
                    break
                midi_note_index += 1
                midi_note_flag = 1
                note.append(notes[midi_note_index])
                curr_note_interval += notes[midi_note_index][1]
            else:
                break
        
        mappings.append((note, all_phonemes, all_wav_data, output_rate))
        syllable_index += 1
        midi_note_index += 1
        pass
    
    return mappings
    pass

# Part 2: Adjust MIDI tonality according to the average pitch of speech.
def midi_key_shift(speech_mean_f0, mappings):
    """
    MIDI as a whole is shifted based on the average F0 of Speech
    @params
    mappings: mapping list consists of note list, phoneme list, start and end.
    speech_mean_pitch: the average pitch of speech. This average is the average of all the non-zero values.
    @return
    output_mappings: The list of mapping relations after overall toning consists of note list, phoneme list, start and end.
    notes_mean_pitch: average pitch of notes in mapping
    """
    
    # extract all pitches
    tar_pitch_midi = []
    for map in mappings:
        for note in map[0]:
            tar_pitch_midi.append(note[0])
    
    speech_mean_f0_midi = hz2midi(speech_mean_f0)
    notes_mean_pitch = round(sum(tar_pitch_midi)/len(tar_pitch_midi))
    
    # transpose number
    trans = notes_mean_pitch - speech_mean_f0_midi
    
    output_mappings = []
    # transpose the notes
    for map in mappings:
        # new empty tuple
        n = []

        for note in map[0]:
            n.append((int(note[0]-trans), note[1], note[2]))
        
        output_mappings.append((n, map[1], map[2], map[3]))
  
    return output_mappings, notes_mean_pitch

# Part 3: Adjust pitch to get pitch-augmented wav.
def pitch_shift(mappings, ori_wav, fs, frame_period=12.5):
    """
    Adjust the pitch according to the mapping
    @params
    mappings: mapping list from note_syllable_mapping
    ori_wav: Raw wav data
    fs: sampling rate
    @return
    output_wav: wav data after pitch shift
    """
    
    # load files
    x = ori_wav
    x = x.astype(np.double)

    _f0, t = pw.dio(x, fs, frame_period=frame_period)    # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
    y = f0                            # for pitch adjusments
    
    silent_mask = np.where(f0==0)[0]
    i = 0
    for map in mappings:
        # editing pitch
        # one-to-one
        if len(map[0]) == 1:
            for i in range(len(map[1])):
                start_mel = map[2][i][0]
                end_mel = map[2][i][1]
                y[start_mel:end_mel] = midi2hz(map[0][0][0])
        
        
        # many-to-one
        # syllable structure:
        #   (consonant*)(vowel)(consonant*)
        # syllable mapping rules:
        #   1. all cosonants before vowel will be assigned to the first note
        #   2. all the notes will having equal length based on the total mels vowel
        #   3. all cosonants before vowel will be assigned to the last note      
        else:
            vowel_mask = np.zeros(len(map[1]))
            for i, phone in enumerate(map[1]):
                if isVowel(phone):
                    vowel_mask[i] = 1
            vowel_start = map[2][np.where(vowel_mask==1)[0][0]][0] # first vowel start mel
            vowel_end = map[2][np.where(vowel_mask==1)[0][-1]][1] # last vowel end mel
            
            # length of every note will be sum(vowel_mel_length)/len(vowel)
            avg_len = (vowel_end - vowel_start) / len(map[0])
            
            flag = 0 # phase of editing many-to-one
            for i, mask in enumerate(vowel_mask):
                if flag == 0 and mask == 1:
                    flag += 1
                elif flag == 1 and mask == 0:
                    flag += 1
                    
                if flag == 0:
                    y[map[2][i][0]:map[2][i][1]] = midi2hz(map[0][0][0])
                
                if flag == 1:
                    for k in range(len(map[0])):
                        y[int(vowel_start + k*avg_len):int(vowel_start + (k+1)*avg_len)] = midi2hz(map[0][k][0])
                        
                if flag == 2:
                    y[map[2][i][0]:map[2][i][1]] = midi2hz(map[0][-1][0])
    
    # keeping silent mels silent
    for sil in silent_mask:
        y[sil] = 0
                        
    # finetune 
    female_like_sp = np.zeros_like(sp)
    for f in range(female_like_sp.shape[1]):
        female_like_sp[:, f] = sp[:, int(f/1.3)]

    female_like = pw.synthesize(y, female_like_sp, ap, fs, frame_period=frame_period)

    return female_like
    
# Part 4: Adjust duration to get duration-augmented wav.
def duration_change(mappings, ori_wav, sr):
    """
    Adjust the pitch length according to the mapping
    @params
    mappings: mapping list from note_syllable_mapping
    ori_wav: raw wav data
    @return
    output_wav: wav data after duration change
    """
    INTERVAL = 12.5
    
    relation_index = 0
    
    secret_string = ''.join(secrets.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(12))

    # adjust duration of each note
    for mapping in mappings:
        note, phoneme_list, wav_data, output_rate = mapping
        
        wav_start = wav_data[0][0]
        wav_end = wav_data[len(wav_data) - 1][1]

        sf.write(secret_string + 'cut' + str(relation_index) + '.wav', ori_wav[int(wav_start*INTERVAL/1000*sr):int(wav_end*INTERVAL/1000*sr)], sr, 'PCM_24')
        os.system("ffmpeg -loglevel quiet -n -i " + secret_string + 'cut' + str(relation_index) + '.wav' + " -filter:a " + "atempo=" + str(output_rate) + " " + secret_string + 'temp_cut' + str(relation_index) + '.wav')
        relation_index += 1
        pass

    output_wav = []
    
    for index in range(0, relation_index):
        y, sr = librosa.load(secret_string + 'temp_cut' + str(index) + '.wav', sr)
        output_wav = np.hstack((output_wav, y))

    os.system("rm -rf " + secret_string + "*")
    
    return output_wav

def main():
    # metadata of libritts dataset
    frame_period = 12.5
    meta_data = pd.read_csv(metadata_dir)
    meta_datas = read_meta_data(meta_data)

    # process using multithreading
    def worker(meta_data):
        path, wave_name, phone, new_phone = meta_data
        
        while True:
            s_midi_path = random.choice(all_midi_path)
            midi_path = midi_file_fir + s_midi_path
            notes = midi2notes(midi_path)

            if len(notes) > len(new_phone):
                break
            pass

        s_midi_path = s_midi_path.split(".")[0]

        try:
            wav, sr = librosa.core.load(path, sr=None)
            syllables = get_syllables(mel_data[wave_name], phone, new_phone)
            
            # Part 1: Determine the correspondence between notes and syllables (one-to-many or many-to-one) according to the duration of MIDI and speech.
            mappings = note_syllable_mapping(notes, syllables)
            x = wav.astype(np.double)
            _f0, t = pw.dio(x, sr, frame_period=frame_period)    # raw pitch extractor
            f0 = pw.stonemask(x, _f0, t, sr)  # pitch refinement
            nonz = np.nonzero(f0)
            mean_f0 = 0
            for index in nonz:
                mean_f0 = mean_f0 + f0[index] 
            
            # Part 2: Adjust MIDI tonality according to the average pitch of speech.
            speech_mean_pitch = sum(mean_f0)/len(mean_f0)
            new_mappings, notes_mean_pitch = midi_key_shift(speech_mean_pitch, mappings)
            
            # Part 3: Adjust pitch to get pitch-augmented wav.
            pitch_wav = pitch_shift(new_mappings, wav, sr)
            single_duration_wav = duration_change(new_mappings, wav, sr)

            d_path = os.path.join(output_duration_dir, "/".join(path.split("/")[-4:]))
            p_path = os.path.join(output_pitch_dir, "/".join(path.split("/")[-4:]))
            pd_path = os.path.join(output_pdaugment_dir, "/".join(path.split("/")[-4:]))
            if not os.path.exists(d_path):
                os.makedirs(d_path)
            if not os.path.exists(p_path):
                os.makedirs(d_path)
            if not os.path.exists(pd_path):
                os.makedirs(d_path)
            if not os.path.exists(os.path.join(output_duration_dir, "/".join(path.split("/")[-4:-1]), "-".join(path.split("/")[-3:-1]) + ".trans.txt")):
                os.system("cp " + os.path.join("/".join(path.split("/")[:-1]), "-".join(path.split("/")[-3:-1]) + ".trans.txt") + " " + os.path.join(output_duration_dir, "/".join(path.split("/")[-4:-1]), "-".join(path.split("/")[-3:-1]) + ".trans.txt"))
            
            sf.write(d_path, single_duration_wav, sr, 'PCM_24')
            sf.write(p_path, pitch_wav, sr, 'PCM_24')

            # Part 4: Adjust duration to get duration-augmented wav.
            duration_wav = duration_change(new_mappings, pitch_wav, sr)
            sf.write(pd_path, duration_wav, sr, 'PCM_24')
        except Exception:
            return

    def muli_task(N, tasks):
        pool = multiprocessing.Pool(N)
        pool.map(worker, tasks)
        pool.close()
        pool.join()

    muli_task(number_of_threads, meta_datas)

if __name__ == '__main__':
    pickle_path = "data/pickle/mel_splits.pickle"
    frequency_json_file = 'utils/frequency.json'
    metadata_dir = 'data/speech/phone/dev-clean_metadata.csv'
    dataset_dir = "data/speech/wav/dev-clean"
    midi_file_fir = "data/midis/processed/midi_6tracks"
    output_duration_dir = "data/duration"
    output_pitch_dir = "data/pitch"
    output_pdaugment_dir = "data/pdaugment"
    number_of_threads = 16

    all_midi_path = []

    try:
        pickle_path = sys.argv[1]
        frequency_json_file = sys.argv[2]
        dataset_dir = sys.argv[3]
        midi_file_fir = sys.argv[4]
        metadata_dir = sys.argv[5]
        output_duration_dir = sys.argv[6]
        output_pitch_dir = sys.argv[7]
        output_pdaugment_dir = sys.argv[8]
        number_of_threads = int(sys.argv[9])
    except IndexError:
        print("Need eight command line parameters.")
        # load metadata
        with open(frequency_json_file) as f:
            fre = json.load(f)
        with open(pickle_path, "rb") as f:
            mel_data = pickle.load(f)
        for file in os.listdir("freemidi"):
            if os.path.splitext(file)[1] == '.mid':
                all_midi_path.append(file)
    main()
