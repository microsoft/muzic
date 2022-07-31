data_dir=l2m
user_dir=mass
model=../checkpoints/mass/$1/checkpoint_best.pt # Load Weights

for f in `ls $data_dir`
do
	song_dir=$data_dir/$f/
	fairseq-train $song_dir \
		--user-dir $user_dir \
		--task xmasked_seq2seq \
		--source-langs lyric --target-langs melody \
		--langs lyric,melody \
		--arch xtransformer \
		--mt_steps lyric-melody \
		--attention-dropout 0 --activation-dropout 0 --dropout 0 --adaptive-softmax-dropout 0 --relu-dropout 0 \
		--share-decoder-input-output-embed \
		--attn-outfile $song_dir/attn.txt \
		--reload-checkpoint $model \
		--max-tokens 4096
done
