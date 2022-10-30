DATA_DIR=data-bin/lmd6remi      # Data dir

OMP_NUM_THREADS=$(cat /proc/cpuinfo| grep "processor"| wc -l)

fairseq-validate \
  $DATA_DIR \
  --user-dir museformer \
  --task museformer_language_modeling \
  --path checkpoints/mf-lmd6remi-$1/$2 \
  --batch-size 1 \
  --truncate-test $3 \
  --valid-subset test \
  --num-workers $OMP_NUM_THREADS
