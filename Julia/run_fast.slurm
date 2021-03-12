#!/bin/bash

#SBATCH --job-name=gnr_inference
#SBATCH --partition=HTC050
#SBATCH --ntasks=40
#SBATCH --nodes=1
#SBATCH --nodelist=htc052
#SBATCH --mem=40gb
#SBATCH --time=24:00:00
#SBATCH --output=conv_files.log
#SBATCH --exclude htc033

python Convert_file.py
