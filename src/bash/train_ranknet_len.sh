#!/bin/bash
#
#SBATCH --job-name=ranknet_lin_len1
#SBATCH --output=logs/ranknet_lin1_len_%j.txt  # output file
#SBATCH -e logs/ranknet_lin1_len%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:2
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=60000
#
#SBATCH --ntasks=1

python -u train_ranknet.py tfidf_length model_tfidf_length.pt
