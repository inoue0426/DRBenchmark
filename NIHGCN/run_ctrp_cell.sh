#!/bin/bash

#SBATCH --job-name='CtrC NIH'
#SBATCH --mem=64G
#SBATCH --partition=a100-4 
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

module load conda
conda activate /scratch.global/$USER/myenvfile.d/mamba.sh
conda activate genex
python New_Cell_ctrp.py
