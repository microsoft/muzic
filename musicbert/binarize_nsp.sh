#!/bin/bash
# 

[[ -z "$1" ]] && { echo "PREFIX is empty" ; exit 1; }
PREFIX=$1
[[ -d "${PREFIX}_data_bin" ]] && { echo "output directory ${PREFIX}_data_bin already exists" ; exit 1; }
fairseq-preprocess \
    --only-source \
    --trainpref ${PREFIX}_data_raw/train.txt \
    --validpref ${PREFIX}_data_raw/test.txt \
    --destdir ${PREFIX}_data_bin/input0 \
    --srcdict ${PREFIX}_data_raw/dict.txt \
    --workers 24
fairseq-preprocess \
    --only-source \
    --trainpref ${PREFIX}_data_raw/train.label \
    --validpref ${PREFIX}_data_raw/test.label \
    --destdir ${PREFIX}_data_bin/label \
    --workers 24
