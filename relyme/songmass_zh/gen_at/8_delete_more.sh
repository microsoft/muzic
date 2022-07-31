data_dir=l2m
for idx in `ls $data_dir`
do
	folder=$data_dir/$idx
	rm $folder/dict.*
	rm $folder/*.idx
	rm $folder/*.bin
done
