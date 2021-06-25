#!/bin/bash

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=4000
PEAK_LR=0.00005
TOKENS_PER_SAMPLE=8192
MAX_POSITIONS=8192
BATCH_SIZE=64
MAX_SENTENCES=4
N_GPU_LOCAL=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
UPDATE_FREQ=$((${BATCH_SIZE} / ${MAX_SENTENCES} / ${N_GPU_LOCAL}))
if [ -z ${1+x} ]; then echo "subset not set" && exit 1; else echo "subset = $1"; fi
if [ -z ${2+x} ]; then echo "num classes not set" && exit 1; else echo "num classes = $2"; fi
if [ -z ${3+x} ]; then echo "fold index not set" && exit 1; else echo "fold index = $3"; fi
if [ -z ${4+x} ]; then echo "model not set" && exit 1; else echo "model = $4"; fi
HEAD_NAME=${1}_head
CHECKPOINT_SUFFIX=genre_${1}_${3}_$(basename ${4%.pt})
MUSICBERT_PATH=${4}

fairseq-train ${1}_data_bin/${3} --user-dir musicbert \
    --restore-file $MUSICBERT_PATH \
    --max-update $TOTAL_NUM_UPDATES \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-positions $MAX_POSITIONS \
    --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES})) \
    --task sentence_prediction_multilabel \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --num-workers 0 \
    --init-token 0 --separator-token 2 \
    --arch musicbert_${CHECKPOINT_SUFFIX##*_} \
    --criterion sentence_prediction_multilabel \
    --classification-head-name $HEAD_NAME \
    --num-classes $2 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --log-format simple --log-interval 100 \
    --best-checkpoint-metric f1_score_micro --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --checkpoint-suffix _${CHECKPOINT_SUFFIX} \
    --no-epoch-checkpoints \
    --find-unused-parameters
