#!/bin/bash

# Ensure the output directory exists
data_dir=${1:-"data_short/"}
mono_data_dir=$data_dir/mono/
para_data_dir=$data_dir/para/
save_dir=$data_dir/processed/

# set this relative path of MASS in your server
user_dir=${2:-"mass"}

mkdir -p $data_dir $save_dir $mono_data_dir $para_data_dir


# Generate Monolingual Data
for lg in lyric melody
do
	fairseq-preprocess \
		--task cross_lingual_lm \
		--srcdict $mono_data_dir/dict.$lg.txt \
		--only-source \
		--trainpref $mono_data_dir/train --validpref $mono_data_dir/valid \
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
	--source-lang lyric --target-lang melody \
	--trainpref $para_data_dir/train --validpref $para_data_dir/valid \
	--destdir $save_dir \
	--srcdict $para_data_dir/dict.lyric.txt \
	--tgtdict $para_data_dir/dict.melody.txt

fairseq-preprocess \
	--user-dir $user_dir \
	--task xmasked_seq2seq \
	--source-lang melody --target-lang lyric \
	--trainpref $para_data_dir/train --validpref $para_data_dir/valid \
	--destdir $save_dir \
	--srcdict $para_data_dir/dict.melody.txt \
	--tgtdict $para_data_dir/dict.lyric.txt

