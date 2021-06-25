python train.py \
  --device '0,1,2,3' \
  --num_pieces 1 \
  --model_config 'config/model_config_small.json' \
  --pretrained_model '' \
  --model_dir 'model' \
  --root_path 'data/lyrics/' \
  --raw_data_dir 'lyrics_22w' \
  --batch_size 8 \
  --epochs 1 \
  --enable_final \
  --enable_sentence \
  --enable_relative_pos \
  --reverse \
  --model_sign '1a' \
  --tokenize \
  --raw   

