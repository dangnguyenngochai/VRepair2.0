#!/bin/bash

python /home/lgm/VRepair2.0/src/generate_transformer_sweep_config.py -train_features_file="/home/lgm/VRepair2.0/vul_data/random_fine_tune_train.src.txt" \
-train_labels_file="/home/lgm/VRepair2.0/vul_data/random_fine_tune_train.tgt.txt" \
-eval_features_file="/home/lgm/VRepair2.0/vul_data/random_fine_tune_valid.src.txt" \
-eval_labels_file="/home/lgm/VRepair2.0/vul_data/random_fine_tune_valid.tgt.txt" \
-sweep_root_path="/home/lgm/VRepair2.0/param_sweep_tgt/"
