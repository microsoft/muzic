# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License. 
#

[[ -z "$1" ]] && { echo "PREFIX is empty" ; exit 1; }
PREFIX=$1
[[ -d "${PREFIX}_data_bin" ]] && { echo "output directory ${PREFIX}_data_bin already exists" ; exit 1; }
for i in {0..4}
do
	echo "Proccessing fold $i"
	mkdir -p ${PREFIX}_data_bin/$i
	fairseq-preprocess \
	    --only-source \
	    --trainpref ${PREFIX}_data_raw/$i/train.txt \
	    --validpref ${PREFIX}_data_raw/$i/test.txt \
	    --destdir ${PREFIX}_data_bin/$i/input0 \
	    --srcdict ${PREFIX}_data_raw/$i/dict.txt \
	    --workers 24
	fairseq-preprocess \
	    --only-source \
	    --trainpref ${PREFIX}_data_raw/$i/train.label \
	    --validpref ${PREFIX}_data_raw/$i/test.label \
	    --destdir ${PREFIX}_data_bin/$i/label \
	    --workers 24
	cp ${PREFIX}_data_raw/$i/train.label ${PREFIX}_data_bin/$i/label/train.label
	cp ${PREFIX}_data_raw/$i/test.label ${PREFIX}_data_bin/$i/label/valid.label
done
echo "Done"
