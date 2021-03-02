#!/bin/bash

#SBATCH --job-name=gnr_inference
#SBATCH --partition=highPrio
#SBATCH --ntasks=40
#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/Single_Up_70_0.log
#SBATCH --exclude htc033

python MRNA_Single_UpRegulation.py
