import os

datasets = ["ctrp", "nci", "gdsc1", "gdsc2"]
targets = ["cell", "drug"]

base_script = """#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --mem=40gb
#SBATCH --requeue
#SBATCH --job-name='mofgcn_{data}_{target}'
#SBATCH --partition=msigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --account=kuangr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --output=logs/mofgcn_{data}_{target}_%j.out
#SBATCH --error=logs/mofgcn_{data}_{target}_%j.err

module load conda
source activate genex

python run_mofgcn.py --data {data} --target {target} --n-jobs 12
"""

os.makedirs("jobs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for data in datasets:
    for target in targets:
        script_name = f"jobs/mofgcn_{data}_{target}.sbatch"
        with open(script_name, "w") as f:
            f.write(base_script.format(data=data, target=target))

print("✅ すべての .sbatch ファイルを jobs/ に生成しました。")

