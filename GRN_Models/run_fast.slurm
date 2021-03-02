#!/bin/bash

#SBATCH --job-name=gnr_inference
#SBATCH --partition=HTC050
#SBATCH --ntasks=40
#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --time=24:00:00
#SBATCH --output=ssa.log
#SBATCH --exclude htc033

python MRNA_Single_UpRegulation.py
