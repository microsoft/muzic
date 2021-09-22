#!/bin/bash

data_dir=${1:-"data_org"}

mkdir -p $data_dir/mono

cp data/dict.*.txt $data_dir/mono/
cp data/dict.*.txt $data_dir/para/

cat $data_dir/para/train.lyric | sed '/^$/d' > $data_dir/mono/train.lyric
cat $data_dir/para/train.melody | sed '/^$/d' > $data_dir/mono/train.melody
cat $data_dir/para/valid.lyric | sed '/^$/d' > $data_dir/mono/valid.lyric
cat $data_dir/para/valid.melody | sed '/^$/d' > $data_dir/mono/valid.melody
