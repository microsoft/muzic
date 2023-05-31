# 运行attribute2music的数据
start=${1}
end=${2}
#input args
model_size="xl"
k=15
command_name=${3}
need_num=64

temp=1.0
ngram=0


datasets_name="truncated_5120"
checkpoint_name="checkpoint_best"
BATCH_SIZE=8

# models parameters
model_name="linear_mask-${model_size}-truncated_5120"


cd linear_mask

DATA_DIR="../data/${datasets_name}"
checkpoint_path="../checkpoints/${model_name}/${checkpoint_name}.pt"
ctrl_command_path="../data/infer_input/${command_name}.bin"
save_root="../generation/${model_name}-${checkpoint_name}/${command_name}/topk${k}"



mkdir -p ${save_root}
echo "generating from ${checkpoint_path}"
echo "save to ${save_root}"


python -u interactive_dict_v5.py \
${DATA_DIR}/data-bin \
--task language_modeling_control \
--path $checkpoint_path \
--ctrl_command_path $ctrl_command_path \
--save_root $save_root \
--need_num ${need_num} \
--start ${start} \
--end ${end} \
--max-len-b 5120 \
--min-len 512 \
--sampling \
--beam 1 \
--sampling-topk ${k} \
--temperature ${temp} \
--no-repeat-ngram-size ${ngram} \
--buffer-size ${BATCH_SIZE} \
--batch-size ${BATCH_SIZE}

