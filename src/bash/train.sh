#!/bin/bash
#
#SBATCH --job-name=lstm
#SBATCH --output=logs/lstm_%j.txt  # output file
#SBATCH -e logs/lstm_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=50000
#
#SBATCH --ntasks=1

python -u train_lstm.py
