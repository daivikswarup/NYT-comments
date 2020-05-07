#!/bin/bash
#
#SBATCH --job-name=ranknet_bert
#SBATCH --output=logs/ranknet_bert_%j.txt  # output file
#SBATCH -e logs/ranknet_bert_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:4
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=60000
#
#SBATCH --ntasks=1

python -u train_ranknet.py bert model_bert.pt
