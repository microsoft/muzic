# models parameters
datasets_name="Piano"
control_mode="embedding_v2"
feature_num=100
bucket_num=2
command_subset="emotion_rank_100"
rank=0

gpus=1

model_name="${datasets_name}"
MODEL_NAME="${model_name}"


tgt=${1}

checkpoint_name="checkpoint_best"
checkpoint_path="checkpoints/${model_name}/${checkpoint_name}.pt"
command_path="data/infer_input/inference_command.npy"
save_root="generation/${model_name}-${checkpoint_name}/Q${tgt}"
mkdir -p ${save_root}
export CUDA_VISIBLE_DEVICES=${rank}
echo "generating from ${checkpoint_path} with emotion Q${tgt}!"
python interactive.py \
data/${datasets_name}/data-bin \
--task language_modeling_control \
--path $checkpoint_path \
--ctrl_command_path $command_path \
--save_root $save_root \
--tgt_emotion $tgt \
--need_num 2 \
--max-len-b 1280 \
--min-len 512 \
--sampling \
--beam 1 \
--sampling-topp 0.9 \
--temperature 1.0 \
--buffer-size 2 \
--batch-size 2

