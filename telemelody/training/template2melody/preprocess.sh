#!/bin/bash
#

if [ -z ${1+x} ]; then echo "IN_DIR not set" && exit 1; else echo "IN_DIR = ${1}"; fi
if [ -z ${2+x} ]; then echo "OUT_DIR not set" && exit 1; else echo "OUT_DIR = ${2}"; fi
IN_DIR=$1
OUT_DIR=$2
fairseq-preprocess \
    --source-lang trend --target-lang notes \
    --trainpref data/${IN_DIR}/train \
    --validpref data/${IN_DIR}/valid \
    --testpref data/${IN_DIR}/test \
    --srcdict data/${IN_DIR}/trend.dict.txt \
    --tgtdict data/${IN_DIR}/notes.dict.txt \
    --destdir data-bin/${OUT_DIR} \
    --align-suffix align \
    --workers 24



