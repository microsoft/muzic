#!/bin/bash

data_dir=${1:-"data_org/processed"} # The path of binarized data
user_dir=${2:-"mass"}

fairseq-train $data_dir \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-langs lyric,melody \
  --target-langs lyric,melody \
  --langs lyric,melody \
  --arch xtransformer \
  --mass_steps lyric-lyric,melody-melody \
  --mt_steps lyric-melody,melody-lyric \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --lr 0.00005 --min-lr 1e-09 --warmup-init-lr 1e-07 \
  --criterion label_smoothed_cross_entropy_with_align \
  --attn-loss-weight 1.0 \
  --max-tokens 4096 \
  --dropout 0.1 --relu-dropout 0.1 --attention-dropout 0.1 \
  --max-epoch 20 \
  --max-update 2000000 \
  --share-decoder-input-output-embed \
  --valid-lang-pairs lyric-lyric,melody-melody \
  --no-epoch-checkpoints \
  --skip-invalid-size-inputs-valid-test
