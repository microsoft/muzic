# GETMusic

## 0.Tutorial Video

We make a video to demonstrate how to use GETMusic at this [link](https://www.youtube.com/watch?v=M2TEQF5x6bc)

## 1. Environment

Python environment:
```
pip install -r requirements.txt
```
Click [here](https://1drv.ms/u/s!ArHNvccy1VzPkWGKXZDQY5k-kDi4?e=fFxcEq) to download the model checkpoint.


## 2. Inference

### 2.1 Track Generation

We use the song "Childhood" by Tayu Lo for demonstration. It is located in the "example_data" folder.

To perform track generation, follow these steps:

1. Run the generation script:
```
python track_generation.py --load_path /path-of-checkpoint --file_path example_data/inference
```

2. The script will iterate over each MIDI file in the "example_data/inference" directory, and for each MIDI, you will be prompted to specify composition needs. In this example, the program output is:
```
Resume from /path-of-checkpoint
example_data/inference/childhood.mid
skip?
```

3. Enter any key except 'y' to compose. Then specify the conditional track and the content track:

```
skip?n
Select condition tracks ('b' for bass, 'd' for drums, 'g' for guitar, 'l' for lead, 'p' for piano, 's' for strings, 'c' for chords; multiple choices; input any other key to skip): lc
Select content tracks ('l' for lead, 'b' for bass, 'd' for drums, 'g' for guitar, 'p' for piano, 's' for strings; multiple choices): dgp
```

In this example, we generate drum, guitar, and piano tracks based on the lead and chord tracks. We also truncate the song at a length of 512 to avoid extrapolation.

```
100%|████████| 100/100 [00:06<00:00, 16.52it/s]
sampling, the song has 512 time units
```

The generation process is fast, and you can find the results saved as 'example_data/inference/lc2dgp-childhood.mid'. You can open it with [Musescore](https://musescore.com/) for further composition. 


### 2.2 Advanced Operations

As shown in our demo page, we support the hybrid generation of track-wise composition and infilling. While the specification of such composition needs might seem complicated, we have not found a simpler solution:

1. Run the generation script. 
```
python position_generation.py --load_path /path-of-checkpoint --file_path example_data/inference
```

The script will examine the tracks in the input MIDI and provide a representation visualization and an input example to make the condition specification clear:
```
Resume from /path-of-checkpoint
example_data/inference/childhood.mid
skip?n
The music has {'lead'} tracks, with 865 positions
Representation Visualization:
        0,1,2,3,4,5,6,7,8,...
(0)lead
(1)bass
(2)drum
(3)guitar
(4)piano
(5)string
(6)chord
Example: condition on 100 to 200 position of lead, 300 to 400 position of piano, write command like this:'0,100,200;4,300,400
Input positions you want to condition on:
Input positions you want to empty:
```

2. Here is an example input:
```
Input positions you want to condition on: 0,0,200;6,0,
Input positions you want to empty: 1,0,;4,0,;5,0,
```

In this example, the specified conditions are the first 200 time units of the lead track and the entire chord track. The empty positions include the entire bass, piano, and string tracks. 

### 2.3 Chord Guidance

All the examples mentioned above use chord guidance, which is automatically inferred from the input tracks. If you want to generate tracks from scratch but condition them on chords, the simplest way is to input a song with the desired chord progression and let the model infer the chords.
Unfortunately, we haven't found a user-friendly solution to specify the desired chord progression through interactive input, so we do not open this function code. However, you can modify the code if needed.


## 3. Usage Tips

Here are some tips to enhance your experience with GETMusic:

1.  Check MIDI program ID: GETMusic supports the following instruments (MIDI program): '0': piano, '25':guitar, '32':bass, '48':string, '80':lead melody. You do not need to worry about the percussion program.

2.  About 'bass': if you want to generate a 'bass' track, the default instrument in Musescore is '低音提琴' (Double Bass), which may not sound harmonious. Change it to '原音贝斯' (Electric Bass/Bass Guitar).

3.  Tune the volume: GETScore does not involve volume information. To obtain satisfactory composition results, you may tune the volume of each instrument. For example, our default volume for 'string' may be too loud that covers the lead melody, you may need to turn it down.

4.  Enable Chord Guidance: We recommend always enabling chord guidance when generating music to achieve a regular pattern in the generated music score.
    
5. Incremental generation: Our experience indicates that employing incremental generation when generating multiple tracks from scratch yields improved results in terms of both regularity in music patterns and overall quality. For example, you can conduct a two-stage generation: 

   -   Stage 1: chord -> lead  
   -   Stage 2: chord, lead -> bass, drum, guitar, piano, string

6.  Avoid Domain Gap:

   -   Consider Input Style: GETMusic is trained on a dataset of crawled pop music. If the style of your input MIDI has a significant domain gap from the training data, the generated results may not meet your expectations.
   -   Change Tracks: In some cases, even if your input music is of the pop genre, the generated results may still be unsatisfactory. One possible reason is that: for example, if you want to generate tracks based on your input guitar, and the guitar pattern in your input is more similar to the lead track in the training data, the domain gap appears. In this case, modifying the MIDI program ID of your guitar track to serve as a lead track can help reduce the track-wise domain gap.
   -   Tune the Input: You can try using a different random seed, modify the code to truncate the music length, or add [EMPTY]s in the end of the input to regenerate a result with variations.

## 4. Data Pre-processing and Training

We do not open training data or the data cleansing scripts we used. However, we have included some MIDI files in the "example_data/train" folder to demonstrate data pre-processing:

1. Filter the MIDI files and infer chords. This process may take a while. The processed MIDIs will be saved in the [OctupleMIDI](https://github.com/microsoft/muzic/tree/main/musicbert) format in the "example_data/processed_train/oct.txt" file. Run the following command:
```
python preprocess/to_oct.py example_data/train example_data/processed_train
```

The output will be:
```
SUCCESS: example_data/train/0_10230_TS0.mid
SUCCESS: example_data/train/0_10232_TS0.mid
SUCCESS: example_data/train/0_10239_TS0.mid
SUCCESS: example_data/train/0_01023_TS0.mid
4/4 (100.00%) MIDI files successfully processed
```

2. Construct the vocabulary by running the following command:
```
python preprocess/make_dict.py example_data/processed_train/ 3
```

The number 3 indicates that only tokens appearing more than 3 times should be included in the vocabulary. The output will display the tokens details in each track. Use the last two rows of the output to modify the last two rows in 'getmusic/utils/midi_config.py' as follows:
```
tracks_start = [16, 144, 272, 408, 545, 745]
tracks_end = [143, 271, 407, 544, 744, 903]
```

3. Configure the training and valid sets by running the following command:
```
python preprocess/binarize.py example_data/processed_train/pitch_dict.txt example_data/processed_train/oct.txt example_data/processed_train
```

The output will display the number of files in the validation and train sets:
```
# valid set: 10
100%|██████| 10/10 [00:00<00:00, 1227.77it/s]
valid set has 10 reps
| #train set: 9
100%|██████| 9/9 [00:00<00:00, 1317.12it/s]
train set has 9 reps
```

After executing these commands, you will find the following files in the "processed_train" folder:
```
|example_data
|----processed_train
|--------oct.txt
|--------pitch_dict.txt
|--------train_length.npy
|--------train.data
|--------train.idx
|--------valid_length.npy
|--------valid.data
|--------valid.idx
```

4. Before training the model, modify the "config/train.yaml" file as follows:
   - In line 14, 66, and 72, change the value of 'vocab_size' to the number of tokens in "pitch_dict.txt" plus 1 (Added 1 is for \[EMPTY\], and you do not need to worry about \[MASK\] since it is considered in the code).
   - In line 25, change the value of 'vocab_path' to the path of the vocabulary file, which in this example is "example_data/processed_train/pitch_dict.txt".
   - In lines 65 and 71, change the value of 'data_folder' to the name of the data folder, which in this example is "example_data/processed_train".

   Modify other parameters such as the scheduler, optimizer, batch size, etc., as per your requirements.

5. Finally, train the model by running the following command:
```
python train.py
```

### 5.Acknowledgement

We appreciate to the following authors who make their code available:

1. [VQ-diffusion](https://github.com/microsoft/VQ-Diffusion/tree/e227b2643f2842d562706534cb1c46301e116b1f)

2. [MusicBert](https://github.com/microsoft/muzic/tree/main/musicbert)

3. [Roformer (Su et al.) and transformers](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/roformer/modeling_roformer.py)

