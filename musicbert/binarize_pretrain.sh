#!/bin/bash
[[ -z "$1" ]] && { echo "PREFIX is empty" ; exit 1; }
PREFIX=$1
[[ -d "${PREFIX}_data_bin" ]] && { echo "output directory ${PREFIX}_data_bin already exists" ; exit 1; }
fairseq-preprocess \
    --only-source \
    --srcdict ${PREFIX}_data_raw/dict.txt \
    --trainpref ${PREFIX}_data_raw/midi_train.txt \
    --validpref ${PREFIX}_data_raw/midi_valid.txt \
    --testpref ${PREFIX}_data_raw/midi_test.txt \
    --destdir ${PREFIX}_data_bin \
    --workers 24
