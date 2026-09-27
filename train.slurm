#!/bin/bash
#SBATCH --job-name=progen-train
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# --mail-type/--mail-user: uncomment to get an email when the job ends
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=jeff.zw07@gmail.com

set -euo pipefail

module load anaconda3/2024.10-1
conda activate progen

# train.py saves to models/ and inference.py writes to design/; torch.save
# fails at the END of epoch 1 if models/ is missing, which wastes the run.
mkdir -p models design logs

nvidia-smi
python -u train.py
