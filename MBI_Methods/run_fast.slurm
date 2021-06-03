#!/bin/bash

#SBATCH --job-name=gnr_inference
#SBATCH --partition=HTC050
#SBATCH --ntasks=40
#SBATCH --nodes=1
%#SBATCH --nodelist=htc051
#SBATCH --mem=40gb
#SBATCH --time=78:00:00
#SBATCH --output=Batch05.log
#SBATCH --exclude htc033

python Batch_Run.py
