# Ensure the output directory exists
data_dir=l2m
user_dir=mass
lrc_dict=/tf/music_align_chn/data/para/dict.lyric.txt
mld_dict=/tf/music_align_chn/data/para/dict.melody.txt

for idx in `ls $data_dir`
do
	process_dir=$data_dir/$idx
	save_dir=$process_dir
	mkdir -p $save_dir
	fairseq-preprocess \
		--user-dir $user_dir \
        	--task xmasked_seq2seq \
        	--source-lang lyric --target-lang melody \
        	--trainpref $process_dir/train \
		--destdir $save_dir \
        	--srcdict $lrc_dict \
        	--tgtdict $mld_dict
done

