# models parameters
datasets_name="TopMAGD"
control_mode="embedding_v2"
feature_num=100
bucket_num=2
command_subset="emotion_rank_100"
rank=0

gpus=1

WARMUP_UPDATES=16000     # Warmup the learning rate over this many updates
PEAK_LR=1e-4            # Peak learning rate, adjust as needed

model_name="${datasets_name}-${control_mode}-bucket${bucket_num}-${command_subset}"
MODEL_NAME="${model_name}"


echo "training the model: ${MODEL_NAME}!"

BATCH_SIZE=1 # BATCH
UPDATE_FREQ=1

DATA_DIR=data/${datasets_name}     # Data dir

export MKL_THREADING_LAYER=GNU # for "import numpy" "import torch" order bug



OMP_NUM_THREADS=$(cat /proc/cpuinfo| grep "processor"| wc -l)
let "port_rank=$rank+666"
#python -m torch.distributed.launch \
#--nproc_per_node=${gpus} \
#--master_port=${port_rank} \
python train.py  \
$DATA_DIR/data-bin \
--truncated_length 2560 \
--task language_modeling_control \
--arch linear_transformer_lm_std \
--control_mode $control_mode \
--command_path $DATA_DIR \
--feature_num $feature_num \
--bucket_num $bucket_num \
--sample-break-mode eos \
--tokens-per-sample 10000000 \
--max-tokens 10000000 \
--batch-size $BATCH_SIZE \
--batch-size-valid $BATCH_SIZE \
--update-freq $UPDATE_FREQ \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--adam-eps 1e-9 \
--weight-decay 0.01 \
--lr $PEAK_LR \
--lr-scheduler inverse_sqrt \
--warmup-updates $WARMUP_UPDATES \
--log-format simple \
--log-interval 10 \
--tensorboard-logdir tb_log/$MODEL_NAME  \
--num-workers "$OMP_NUM_THREADS" \
--max-update 600000 \
--validate-interval 2 \
--save-interval-updates 2000 \
--save-dir checkpoints/$MODEL_NAME \
--no-epoch-checkpoints  \
--find-unused-parameters \
--patience 20