#!/bin/bash
lyric=lyric/lyric.txt
dict_path=data_short/para/
data_dir=test_data
gen_at_dir=gen_at
dst=l2m_midi

stage=5
# data preparation
if [ $stage -le 0  ]
then
    echo "------------------------"
    echo "Step0: Data preparation ... "
    echo "------------------------"
    rm -r ${data_dir}
    mkdir ${data_dir}
    mkdir ${data_dir}/mono
    mkdir ${data_dir}/para
    mkdir ${data_dir}/processed

    python songmass_lyric_convert.py ${lyric} lines ${data_dir}/mono
    # mv valid.* ${data_dir}/mono
    # mv song_id.txt ${data_dir}/para
fi

if [ $stage -le 1 ]
then
    echo "------------------------"
    echo "Step1: Copy data ... "
    echo "------------------------"
    cp ${dict_path}/dict.lyric.txt ${data_dir}/mono
    cp ${dict_path}/dict.melody.txt ${data_dir}/mono
    cp ${data_dir}/mono/valid.lyric ${data_dir}/mono/train.lyric
    cp ${data_dir}/mono/valid.melody ${data_dir}/mono/train.melody
    cp ${data_dir}/mono/* ${data_dir}/para
    cp ${data_dir}/mono/* ${data_dir}/processed
fi

# preprocessing data
if [ $stage -le 2 ]
then
    echo "------------------------"
    echo "Step2: Preprocessing data ... "
    echo "------------------------"
    sh l2m_preprocess.sh ${data_dir}
fi

# infer
if [ $stage -le 3 ]
then
    echo "------------------------"
    echo "Step3: Infer"
    echo "------------------------"
    rm -r result
    mkdir result
    for i in {1..5}
    do
        sh l2m_pr_align_score.sh > result/l2m_test_"${i}"
    done
fi

# gen_align
if [ $stage -le 4 ]
then
    echo "------------------------"
    echo "Step4: Generate Align File"
    echo "------------------------"
    cd ${gen_at_dir}
    for i in {1..5}
    do
        sh run_all.sh ../result/l2m_test_"${i}" pretrain_align ../${data_dir} $i
    done
    cd ../
fi

# gen_midi
if [ $stage -le 5 ]
then
    echo "------------------------"
    echo "Step5: Generate MIDI"
    echo "------------------------"
    # rm -r ../l2m_merge
    # cp -r l2m_merge ../
    # cd ../

    # get midi without lyrics
    for i in {1..5}
    do
        python generate_melody_songmass.py gen_at/l2m_merge_"${i}"
    done

    rm -r ${dst}
    mkdir ${dst}
    for i in {1..5}
    do  
        mkdir ${dst}/midi_"${i}"
        python convert_midi_songmass_en.py gen_at/l2m_merge_"${i}" ${dst}/midi_"${i}"
    done
fi

# ranking
if [ $stage -le 6 ]
then
    echo "------------------------"
    echo "Step6: Ranking"
    echo "------------------------"

    python ranking.py ${dst} ./best_midi
fi
