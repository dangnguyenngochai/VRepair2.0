#!/bin/bash

onmt_translate -gpu 0 \
		-batch_size 4 \
		-model model_step_20000.pt \
		-src /home/lgm/VRepair2.0/vul_data/random_fine_tune_valid.src.txt \
		-tgt /home/lgm/VRepair2.0/vul_data/random_fine_tune_valid.tgt.txt \
		-output predictions_for_best_model.txt \
		-verbose
