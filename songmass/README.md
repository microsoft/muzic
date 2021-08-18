# SongMASS

* The paper: [SongMASS: Automatic Song Writing with Pre-training and Alignment Constraint](https://arxiv.org/abs/2107.01875)

## Requirements
The requirements for running SongMASS are listed in `songmass/requirements.txt`. To install the requirements, run:
```bash
pip install -r requirements.txt
```


## Data 
We obtain LMD dataset from [here](https://github.com/yy1lab/Lyrics-Conditioned-Neural-Melody-Generation). We privode a [script](data/generate_lmd_dataset) to parse LMD data in our experiments. We provide a example to instruct how to parse LMD data in our paper.

```bash
git clone https://github.com/yy1lab/Lyrics-Conditioned-Neural-Melody-Generation
DATADIR=Lyrics-Conditioned-Neural-Melody-Generation/lmd-full_MIDI_dataset/Sentence_and_Word_Parsing
OUTPUTDIR=data_org

cd data
python generate_lmd_dataset.py --lmd-data-dir $DATA_DIR --output-dir $OUTPUTDIR
```
Based on the above scripts, data samples will be generated under `data_org` directory as follow:

```bash
├── data_org
│   └── para
│        ├── train.melody
│        ├── train.lyric
│        ├── valid.melody
│        ├── valid.lyric
│        ├── test.melody
│        ├── test.lyric
│        ├── song_id_valid.txt
│        └── song_id_test.txt
```
We can consider para data as mono data and convert lyric file into bpecode to handle dictionary. 


## Training
TODO


## Inference
TODO

## Evaluation
We provide scripts under the [evaluation](evaluate/) folder to test the pitch/duration similarity and melody distance. The examples are as:
```bash
LYRIC={"The lyric file of ground truth"}
MELODY={"The melody file of ground truth"}
HYPOS={"The generated result in fairseq format"}
SONG_ID={"SONG ID FILE"}

cd evaluate/

# pitch distribution similarity 
python evaluate_histo.py \
  --lyric-file $LYRIC \
  --melody-file $MELODY \
  --song-id-file $SONG_ID \
  --generated-melody-file $HYPOS \
  --metric pitch 

# duration distribution similarity
python evaluate_histo.py \
  --lyric-file $LYRIC \
  --melody-file $MELODY \
  --song-id-file $SONG_ID \
  --generated-melody-file $HYPOS \
  --metric duration  
  
# melody distance
python evaluate_timeseries.py \
  --lyric-file $LYRIC \
  --melody-file $MELODY \
  --song-id-file $SONG_ID \
  --generated-melody-file $HYPOS
```
