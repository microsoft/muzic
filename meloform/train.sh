#!/bin/bash

data_dir=${1:-"./data/train/processed/processed_para/"} # The path of binarized data
user_dir=${2:-"meloform"}
task=${3:-"meloform_task"}
arch=${4:-"transformer_meloform"}
out_dir=${5:-"checkpoints"}
lr_min=${6:-"1e-07"}
lr_max=${7:-"1e-04"}
criterion=${8:-"meloform_label_smoothed_cross_entropy_with_alignment"}

fairseq-train --task ${task} \
        $data_dir \
        --user-dir ${user_dir} \
        --arch $arch \
        --update-freq 1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --criterion ${criterion} --label-smoothing 0 \
        --lr-scheduler inverse_sqrt --warmup-init-lr $lr_min --warmup-updates 4000 --lr $lr_max \
        --encoder-embed-dim 256 --decoder-embed-dim 256 \
        --encoder-attention-heads 4 --decoder-attention-heads 4 \
        --encoder-layers 4 --decoder-layers 4 \
        --dropout 0.2 --weight-decay 0.0001 \
        --max-update 5000000000 \
        --save-dir ${out_dir} \
        --tensorboard-logdir ${out_dir}/log \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 4096 --max-source-positions 4096 --max-target-positions 4096 \
        --log-interval 100 --save-interval 1 | tee -a ${out_dir}/log.txt
