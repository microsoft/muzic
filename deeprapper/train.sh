#!/bin/bash

python train.py \
  --device '0,1' \
  --stride 1024 \
  --model_config 'config/model_config_small.json' \
  --model_dir 'model' \
  --root_path 'data/lyrics/' \
  --raw_data_dir 'lyrics_samples' \
  --batch_size 1 \
  --epochs 4 \
  --enable_final \
  --enable_sentence \
  --enable_relative_pos \
  --enable_beat \
  --reverse \
  --model_sign 'samples' \
  --with_beat \
  --beat_mode 0 \
  --tokenize \
  --raw  
