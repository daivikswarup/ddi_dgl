#!/bin/bash
#
#SBATCH --job-name=diffpool
#SBATCH --output=logs/logs_diffpool_%j.txt  # output file
#SBATCH -e logs/logs_diffpool_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=40000
#
#SBATCH --ntasks=1

python -u main.py -model Diffpool -no_protiens -savefile diffpool

exit
