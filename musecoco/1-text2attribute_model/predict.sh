python main.py \
    --do_predict \
    --model_name_or_path=CHECKPOINT_PATH \
    --test_file=data/predict.json \
    --attributes=data/att_key.json \
    --num_labels=num_labels.json \
    --output_dir=./tmp \
    --overwrite_output_dir
