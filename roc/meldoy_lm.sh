cat data/lmd_matched/lib-maj.notes data/lmd_matched/lib-min.notes > data/lmd_processed/all.notes
shuf all.notes | split -a1 -d -l $(( $(wc -l <all.notes) * 90 / 100 )) - output
mv output0 train.notes
mv output1 valid.notes
fairseq-preprocess --only-source \
              --task language_modeling \
              --trainpref data/lmd_processed/train.notes  \
              --validpref data/lmd_processed/valid.notes   \
              --destdir data/lmd_processed   \
              --workers 40


mkdir music-ckps
fairseq-train data/lmd_processed/   \ 
            --arch transformer_lm   \
            --task language_modeling     \
            --decoder-attention-heads 4    \
            --decoder-embed-dim 256  --decoder-input-dim 256    \
            --decoder-output-dim 256  --decoder-layers 4   \
            --update-freq 1  --optimizer adam --adam-betas '(0.9, 0.98)' \
            --adam-eps 1e-6 --clip-norm 0.0  --criterion label_smoothed_cross_entropy \
            --label-smoothing 0.1  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
            --warmup-updates 4000 --lr 0.0001  --attention-dropout 0.1  \   
            --dropout 0.1 --weight-decay 0.01  --max-update 50000 --save-dir music-ckps2 \
            --batch-size 1  --max-target-positions 512  --log-interval 100 --patience 20 \
            --no-epoch-checkpoints --best-checkpoint-metric 'ppl' | tee music-ckps/log.txt
