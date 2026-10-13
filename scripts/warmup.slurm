#!/usr/bin/env bash
#SBATCH --job-name=warmup
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/warmup_%j.out
#SBATCH --error=logs/warmup_%j.err

# Warm-up / smoke test on a single GPU.
# Submit from the project root:
#   mkdir -p logs
#   sbatch scripts/warmup.slurm
#   # override args:
#   sbatch scripts/warmup.slurm --limit 20

set -euo pipefail

# Activate the project conda environment.
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate llm-assignment

cd "$(dirname "$0")/.."
echo "host: $(hostname)"
which python
nvidia-smi || true

python scripts/warmup.py "$@"
