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
Based on the above scripts, data samples will be generated under the `data_org` directory. We consider para data as mono data and convert lyric file into bpecode to handle dictionary. The processed bpecode and dictionaries have been uploaded under [data](data/). We move dictionary files to `mono` and `para` directory. The format is as:
```bash
├── data_org
│   └── mono
│        ├── train.melody
│        ├── train.lyric
│        ├── valid.melody
│        ├── valid.lyric
│        ├── dict.lyric.txt
│        └── dict.melody.txt
│   └── para
│        ├── train.melody
│        ├── train.lyric
│        ├── valid.melody
│        ├── valid.lyric
│        ├── test.melody
│        ├── test.lyric
│        ├── dict.lyric.txt
│        ├── dict.melody.txt
│        ├── song_id_valid.txt
│        └── song_id_test.txt
```
We have provide the [script](preprocess.sh) to generate binarized data. The format is as:
```bash
# Ensure the output directory exists
data_dir=data_org/
mono_data_dir=$data_dir/mono/
para_data_dir=$data_dir/para/
save_dir=$data_dir/processed/

# set this relative path of MASS in your server
user_dir=mass

# Generate Monolingual Data
for lg in lyric melody
do
  fairseq-preprocess \
  --task cross_lingual_lm \
  --srcdict $mono_data_dir/dict.$lg.txt \
  --only-source \
  --trainpref $mono_data_dir/train \
  --validpref $mono_data_dir/valid \
  --destdir $save_dir \
  --workers 20 \
  --source-lang $lg
  
  # Since we only have a source language, the output file has a None for the
  # target language. Remove this
  for stage in train valid
  do
    mv $save_dir/$stage.$lg-None.$lg.bin $save_dir/$stage.$lg.bin
    mv $save_dir/$stage.$lg-None.$lg.idx $save_dir/$stage.$lg.idx
  done
done

# Generate Bilingual Data
fairseq-preprocess \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-lang lyric \
  --target-lang melody \
  --trainpref $para_data_dir/train \
  --validpref $para_data_dir/valid \
  --testpref $para_data_dir/test \
  --destdir $save_dir \
  --srcdict $para_data_dir/dict.lyric.txt \
  --tgtdict $para_data_dir/dict.melody.txt

fairseq-preprocess \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-lang melody \
  --target-lang lyric \
  --trainpref $para_data_dir/train \
  --validpref $para_data_dir/valid \
  --testpref $para_data_dir/test \
  --destdir $save_dir \
  --srcdict $para_data_dir/dict.melody.txt \
  --tgtdict $para_data_dir/dict.lyric.txt
```

## Training
```bash
data_dir=data_org/processed # The path of binarized data
user_dir=mass

fairseq-train $data_dir \
  --user-dir $user_dir \
  --save-dir $save_dir \
  --task xmasked_seq2seq \
  --source-langs lyric,melody \
  --target-langs lyric,melody \
  --langs lyric,melody \
  --arch xtransformer \
  --mass_steps lyric-lyric,melody-melody \
  --mt_steps lyric-melody,melody-lyric \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --lr 0.00005 --min-lr 1e-09 \
  --criterion label_smoothed_cross_entropy_with_align \
  --attn-loss-weight 1.0 \
  --max-tokens 4096 \
  --dropout 0.1 --relu-dropout 0.1 --attention-dropout 0.1 \
  --max-epoch 20 \
  --max-update 2000000 \
  --share-decoder-input-output-embed \
  --valid-lang-pairs lyric-lyric,melody-melody \
  --no-epoch-checkpoints \
```

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
