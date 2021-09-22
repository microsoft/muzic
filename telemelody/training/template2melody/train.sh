#!/bin/bash
#

mkdir checkpoints/${1}
fairseq-train data-bin/${1} \
        --arch transformer \
        --update-freq 1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --load-alignments --criterion label_smoothed_cross_entropy_with_alignment --label-smoothing 0 \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 \
        --encoder-embed-dim 256 --decoder-embed-dim 256 \
        --encoder-attention-heads 4 --decoder-attention-heads 4 \
        --encoder-layers 4 --decoder-layers 4 \
        --dropout 0.2 --weight-decay 0.0001 \
        --max-update 500000 \
        --save-dir checkpoints/${1} \
        --max-tokens 4096 --max-source-positions 4096 --max-target-positions 4096 \
        --log-interval 100 --save-interval 10 | tee -a checkpoints/${1}/log.txt
