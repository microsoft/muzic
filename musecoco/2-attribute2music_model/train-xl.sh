model_size="xl"
datasets_name="truncated_5120"

BATCH_SIZE=8 # BATCH

# multi-node distributed training
#Nodes_num=${1}
#gpu_num=${2}


WARMUP_UPDATES=16000     # Warmup the learning rate over this many updates
PEAK_LR=2e-4            # Peak learning rate, adjust as needed
model_name="linear_mask-${model_size}"


MODEL_NAME="${model_name}-${datasets_name}"


echo "training the model: ${MODEL_NAME}!"


UPDATE_FREQ=1

DATA_DIR="data/${datasets_name}"     # Data dir
export MKL_THREADING_LAYER=GNU # for "import numpy" "import torch" order bug

# multi-node distributed training
#export NODE_RANK
#export MASTER_ADDR
#export MASTER_PORT
#export NCCL_DEBUG=info
#echo "Current NODE_RANK: ${NODE_RANK}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}!"

OMP_NUM_THREADS=$(cat /proc/cpuinfo| grep "processor"| wc -l)



#python -u -m torch.distributed.launch --nproc_per_node=${gpu_num} \
#--nnodes=${Nodes_num} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
#--master_port=1234 \
$(which fairseq-train) $DATA_DIR/data-bin \
--user-dir linear_mask \
--task language_modeling_control \
--arch linear_transformer_lm_${model_size} \
--command_path $DATA_DIR \
--truncated_length 5120 \
--command_mask_prob -1 \
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
--max-update 5000000 \
--validate-interval 100000000 \
--validate-interval-updates 5000 \
--save-interval-updates 10000 \
--save-dir checkpoints/$MODEL_NAME \
--no-epoch-checkpoints  \
--find-unused-parameters \
--patience 1000000000000


