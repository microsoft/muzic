MODEL_NAME='mf-lmd6remi-1'

DATA_DIR=data-bin/lmd6remi      # Data dir

# 4 GPUs
UPDATE_FREQ=1

PEAK_LR=5e-4            # Peak learning rate, adjust as needed
WARMUP_UPDATES=16000     # Warmup the learning rate over this many updates

OMP_NUM_THREADS=$(cat /proc/cpuinfo| grep "processor"| wc -l)

ulimit -n 4096

mkdir -p log

fairseq-train \
  $DATA_DIR \
  --user-dir museformer \
  --task museformer_language_modeling \
  --arch museformer_lm_v2s1 \
  --con2con '((((-2, 0), -4, -8, -12, -16, -24, -32),),)' \
  --con2sum '((((None, -32), (-31, -24), (-23, -16), (-15, -12), (-11, -8), (-7, -4), -3,),),)' \
  --num-layers 4 \
  --truncate-train 15360 \
  --truncate-valid 10240 \
  --batch-size 1 \
  --update-freq $UPDATE_FREQ \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --adam-eps 1e-9 \
  --weight-decay 0.01 \
  --lr $PEAK_LR \
  --lr-scheduler inverse_sqrt \
  --warmup-updates $WARMUP_UPDATES \
  --max-update 1000000 \
  --validate-interval 1000000000 \
  --save-interval 1000000000 \
  --save-interval-updates 5000 \
  --fp16 \
  --log-format simple \
  --log-interval 10 \
  --tensorboard-logdir tb_log/$MODEL_NAME  \
  --num-workers "$OMP_NUM_THREADS" \
  --save-dir checkpoints/$MODEL_NAME \
  | tee log/${MODEL_NAME}.log
