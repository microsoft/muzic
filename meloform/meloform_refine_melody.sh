#!/bin/bash

data_dir=${1:-"./data/refine/data_nn"}
song_id=${2:-"0"}
model_path=${3:-"checkpoints/"}
res_dir=${4:-"results/"}
topk=${5:-"-1"}
topp=${6:-"0.9"}
temperature=${7:-"1.0"}
	
python meloform_refine_melody_gen.py $data_dir $model_path checkpoint_best.pt $res_dir $song_id a1 $topk $topp $temperature

python meloform_refine_melody_replace.py $data_dir $song_id b1 a1 $res_dir
python meloform_refine_melody_gen.py $data_dir $model_path checkpoint_best.pt $res_dir $song_id b1 $topk $topp $temperature

echo "$song_id is finished"

