#!/bin/bash
#
#SBATCH --job-name=ranknet_linc
#SBATCH --output=logs/ranknet_linc_%j.txt  # output file
#SBATCH -e logs/ranknet_linc_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:2
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=60000
#
#SBATCH --ntasks=1

python -u train_ranknet.py count model_count.pt
