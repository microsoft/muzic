data_dir=/home/dwj/data/code/model_modify/zhonghao/not_pitch_normalization/evaluate/calculate_ppl/data/l2m
user_dir=/home/dwj/data/code/model_modify/zhonghao/not_pitch_normalization/evaluate/mass
model=/home/dwj/data/code/model_modify/zhonghao/songmass/checkpoints/l2m/checkpoint_last.pt # Load Weights

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
		--criterion label_smoothed_cross_entropy_with_align \
		--prob-outfile $song_dir/prob.txt \
		--reload-checkpoint $model \
		--max-tokens 4096
done
