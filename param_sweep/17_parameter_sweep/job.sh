#!/bin/bash

# Project to run under
#SBATCH -A SNIC2020-5-453
# Name of the job (makes easier to find in the status lists)
#SBATCH -J repair
# Exclusive use when using more than 2 GPUs
#SBATCH --exclusive
# Use GPU
#SBATCH --gres=gpu:k80:1
# the job can use up to x minutes to run
#SBATCH --time=96:00:00

# run the program
onmt_build_vocab -config /home/lgm/VRepair/param_sweep/17_parameter_sweep/vocab_config.yml
sed -i '1iCWE-119\t99999999\nCWE-125\t99999999\nCWE-20\t99999999\nCWE-200\t99999999\nCWE-264\t99999999\nCWE-476\t99999999\nCWE-399\t99999999\nCWE-189\t99999999\nCWE-416\t99999999\nCWE-190\t99999999\nCWE-362\t99999999\nCWE-787\t99999999\nCWE-284\t99999999\nCWE-772\t99999999\nCWE-415\t99999999' /home/lgm/VRepair/param_sweep/17_parameter_sweep/data.vocab.src
onmt_train --config /home/lgm/VRepair/param_sweep/17_parameter_sweep/train_config.yml 2>&1 | tee -a /home/lgm/VRepair/param_sweep/17_parameter_sweep/log.txt

    