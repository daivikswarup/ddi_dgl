#!/bin/bash
#
#SBATCH --job-name=RGCN
#SBATCH --output=logs/logs_RGCN_%j.txt  # output file
#SBATCH -e logs/logs_RGCN_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=40000
#
#SBATCH --ntasks=1

python -u main.py -model RGCN

exit
