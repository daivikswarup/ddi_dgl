#!/bin/bash
#
#SBATCH --job-name=train
#SBATCH --output=logs/logs_noprotien_%j.txt  # output file
#SBATCH -e logs/logs_noprotien_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --mem=40000
#
#SBATCH --ntasks=1

python -u main.py -mincount 20000 -no_protien -savepath ../models/noprotien/

exit
