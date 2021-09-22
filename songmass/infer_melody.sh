#!/bin/bash

data_dir=${1:-"data_org/processed"}
user_dir=${2:-"mass"}
model=${3:-"Model path"}

fairseq-generate $data_dir \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-langs lyric --target-langs melody \
  --langs lyric,melody \
  --source-lang lyric --target-lang melody \
  --mt_steps lyric-melody \
  --gen-subset valid \
  --beam 5 \
  --nbest 5 \
  --remove-bpe \
  --max-len-b 500 \
  --no-early-stop \
  --path $model \
  --sampling
