lrc_dict=/tf/music_align_chn/data/para/dict.lyric.txt
mld_dict=/tf/music_align_chn/data/para/dict.melody.txt

data_dir=l2m
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


















data_dir=m2l
for idx in `ls $data_dir`
do
        process_dir=$data_dir/$idx
        save_dir=$process_dir
        cp $lrc_dict $save_dir/
        cp $mld_dict $save_dir/
	cp $save_dir/train.melody-lyric.lyric.bin $save_dir/valid.melody-lyric.lyric.bin
        cp $save_dir/train.melody-lyric.lyric.idx $save_dir/valid.melody-lyric.lyric.idx
        cp $save_dir/train.melody-lyric.melody.bin $save_dir/valid.melody-lyric.melody.bin
        cp $save_dir/train.melody-lyric.melody.idx $save_dir/valid.melody-lyric.melody.idx
done

