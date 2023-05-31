python main.py \
    --model_name_or_path=bert-large-uncased \
    --do_train \
    --do_eval \
    --train_file=data/test.json \
    --validation_file=data/test.json \
    --attributes=data/att_key.json \
    --max_seq_length=256 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --output_dir=./tmp/ \
    --load_best_model_at_end=True \
    --save_strategy=steps \
    --evaluation_strategy=steps \
    --num_train_epochs=50 \
    --overwrite_output_dir \
    --learning_rate=1e-05 
    # --report_to=wandb \
    # --run_name=T2A