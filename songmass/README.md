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

## Training
TODO


## Inference
TODO

## Evaluation
