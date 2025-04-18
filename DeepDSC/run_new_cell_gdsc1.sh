#!/bin/bash

#SBATCH --job-name='G1C DeepDSC'
#SBATCH --partition gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

module load conda
conda activate /scratch.global/$USER/myenv
python new_cell_gdsc1.py
