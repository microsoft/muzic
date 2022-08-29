#!/bin/bash

data_dir=${1:-"./data/train/processed/"}
para_data_dir=$data_dir/para/
save_dir=$data_dir/processed_para/
user_dir=${2:-"meloform"}
task=${3:-"meloform_task"}	

mkdir -p $data_dir $save_dir $para_data_dir

fairseq-preprocess \
	--user-dir $user_dir \
	--task $task \
	--source-lang template --target-lang melody \
	--trainpref $para_data_dir/train --validpref $para_data_dir/valid --testpref $para_data_dir/test \
	--destdir $save_dir \
	--srcdict $para_data_dir/dict.template.txt \
	--tgtdict $para_data_dir/dict.melody.txt

