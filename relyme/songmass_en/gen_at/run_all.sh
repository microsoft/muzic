echo "L2m result $1"
echo "L2m Model  $2"
echo "Data_dir   $3"

rm -r l2m l2m_merge log
mkdir log

echo "Get Output"
# Seperate Input
python 1_Gen_L2M.py $1 $3
wait

echo "Preprocess"
# Generate Input
sh 3_preprocess_l2m.sh > log/logl2m_gen
wait

echo "Copy"
# Copy dict
sh 5_copy_dict.sh
wait

echo "Model"
# Model
sh 6_l2m_align_attn.sh $2 > log/logl2m_train
wait

echo "Delete"
sh 8_delete_more.sh

echo "Merge"
python 9_l2m_merge_attn.py $3
wait

echo "Generate Output"
python 11_l2m_dp.py

