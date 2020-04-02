#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=logs/test_%j.txt  # output file
#SBATCH -e logs/test_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=80000
#
#SBATCH --ntasks=1

python -u test.py

exit
