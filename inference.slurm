#!/bin/bash
#SBATCH --job-name=progen-design
#SBATCH --partition=GPU-shared
# same sm_75+ requirement as training: no v100 (sm_70) with torch cu128
#SBATCH --gres=gpu:h100-80:1
#SBATCH --ntasks-per-node=5
#SBATCH --time=01:00:00
#SBATCH --output=logs/design_%j.out
#SBATCH --error=logs/design_%j.err

set -euo pipefail

module load anaconda3/2024.10-1
conda activate progen

mkdir -p design logs

# inference.py loads models/best_checkpoint.pt; fail early instead of after the queue wait
test -f models/best_checkpoint.pt || { echo "no checkpoint yet - run train.slurm first"; exit 1; }

nvidia-smi
python -u inference.py
