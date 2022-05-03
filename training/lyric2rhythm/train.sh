#!/bin/bash
#

if [ -z ${1+x} ]; then echo "IN_DIR not set" && exit 1; else echo "IN_DIR = ${1}"; fi
if [ -z ${2+x} ]; then echo "OUT_DIR not set" && exit 1; else echo "OUT_DIR = ${2}"; fi
IN_DIR=$1
OUT_DIR=$2
fairseq-preprocess \
    --source-lang lyric --target-lang beat \
    --trainpref ${IN_DIR}/train \
    --validpref ${IN_DIR}/valid \
    --testpref ${IN_DIR}/test \
    --destdir data-bin/${OUT_DIR} \
    --workers 24

mkdir checkpoints/${2}
fairseq-train data-bin/${2} \
        --arch transformer \
        --update-freq 1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0  --valid-subset valid,test \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 \
        --encoder-embed-dim 256 --decoder-embed-dim 256 \
        --encoder-attention-heads 4 --decoder-attention-heads 4 \
        --encoder-layers 4 --decoder-layers 4 \
        --dropout 0.2 --weight-decay 0.0001 \
        --max-update 500000 \
        --save-dir checkpoints/${2} \
        --max-tokens ${3}  \
        --report-accuracy \
        --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --log-interval 100 --num-workers 10  --no-epoch-checkpoints  | tee -a checkpoints/${2}/log.txt

