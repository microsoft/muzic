lrc_dict=/home/dwj/data/code/model_modify/zhonghao/songmass/data/dict.lyric.txt
mld_dict=/home/dwj/data/code/model_modify/zhonghao/songmass/data/dict.melody.txt

data_dir=m2l
for idx in `ls $data_dir`
do
	process_dir=$data_dir/$idx
	save_dir=$process_dir
	cp $lrc_dict $save_dir/
       	cp $mld_dict $save_dir/
	cp $save_dir/train.lyric-melody.lyric.bin $save_dir/valid.lyric-melody.lyric.bin
	cp $save_dir/train.lyric-melody.lyric.idx $save_dir/valid.lyric-melody.lyric.idx
	cp $save_dir/train.lyric-melody.melody.bin $save_dir/valid.lyric-melody.melody.bin
        cp $save_dir/train.lyric-melody.melody.idx $save_dir/valid.lyric-melody.melody.idx
done
exit

