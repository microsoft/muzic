data_dir=l2m/7/
user_dir=mass
model=../checkpoints/mass/l2m_base_align/checkpoint_best.pt # Load Weights

fairseq-train $data_dir \
	--user-dir $user_dir \
	--task xmasked_seq2seq \
	--source-langs lyric --target-langs melody \
	--langs lyric,melody \
	--arch xtransformer \
	--mt_steps lyric-melody \
	--attention-dropout 0 --activation-dropout 0 --dropout 0 --adaptive-softmax-dropout 0 --relu-dropout 0 \
	--share-decoder-input-output-embed \
	--attn-outfile attn_1.txt \
	--reload-checkpoint $model \
	--max-tokens 4096

