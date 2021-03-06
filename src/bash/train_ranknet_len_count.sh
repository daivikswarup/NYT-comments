#!/bin/bash
#
#SBATCH --job-name=ranknet_lin_len_count
#SBATCH --output=logs/ranknet_lin_len_count_%j.txt  # output file
#SBATCH -e logs/ranknet_lin_len_count_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:2
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=60000
#
#SBATCH --ntasks=1

python -u train_ranknet.py count_length model_count_length.pt
