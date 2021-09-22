#!/bin/bash

data_dir=${1:-"data_org/processed"}
user_dir=${2:-"mass"}
model=${3:-"Model path"}

fairseq-generate $data_dir \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-langs melody --target-langs lyric \
  --langs lyric,melody \
  --source-lang melody --target-lang lyric \
  --mt_steps melody-lyric \
  --gen-subset valid \
  --beam 5 \
  --nbest 5 \
  --remove-bpe \
  --max-len-b 500 \
  --no-early-stop \
  --path $model \
  --sampling
