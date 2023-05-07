#!/bin/bash
export BIG_VUL_PATH=/home/lgm/VRepair/fine_tune_data/big_vul
export VREPAIR_SRC=/home/lgm/VRepair/src
export PREPROCESS_OUTPUT=/home/lgm/VRepair/vul_data

# Extract token data from C source files
python $VREPAIR_SRC/extract.py $BIG_VUL_PATH/commits > $PREPROCESS_OUTPUT/extract.out 2>&1

cat $PREPROCESS_OUTPUT/extract.out

# Generate src/tgt raw files from tokenized data
python $VREPAIR_SRC/gensrctgt.py $BIG_VUL_PATH/commits 3 -meta $BIG_VUL_PATH/commits_metadata.csv >  $PREPROCESS_OUTPUT/gensrctgt.out 2>&1

cat $PREPROCESS_OUTPUT/gensrctgt.out

# Create random split on data
python $VREPAIR_SRC/process_fine_tune_data.py --src_file=$BIG_VUL_PATH/SrcTgt/commits.src.txt \
--tgt_file=$BIG_VUL_PATH/SrcTgt/commits.tgt.txt \
--meta_file=$BIG_VUL_PATH/SrcTgt/commits.meta.txt \
--max_src_length=1000 \
--max_tgt_length=100 \
--generate_random \
--is_big_vul \
--output_dir=$PREPROCESS_OUTPUT/. > $PREPROCESS_OUTPUT/process.out 2>&1

cat $PREPROCESS_OUTPUT/process.out
