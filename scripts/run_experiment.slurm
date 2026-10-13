#!/usr/bin/env bash
#SBATCH --job-name=bench
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

# Run the benchmark matrix (src/run_experiment.py) on one GPU.
#
# TIME LIMIT: 06:00:00 is the maximum wall time allowed by the `normal` QoS,
# which is the only QoS this account is authorised for. That is the real
# ceiling -- do not raise it, the scheduler will reject the job.
#
# STAGED WORKFLOW (variants run one after another, baseline first):
#   mkdir -p logs
#   sbatch --job-name=bench-base scripts/run_experiment.slurm --variants baseline
#   # inspect results/scores/*__baseline.csv, then:
#   sbatch --job-name=bench-impr scripts/run_experiment.slurm --variants improved
#
# The model that dominates runtime is deepseek-r1-distill-qwen-1.5b (it emits a
# long chain of thought), so it is the first candidate for splitting into its
# own job:
#   sbatch scripts/run_experiment.slurm --models deepseek-r1-distill-qwen-1.5b \
#          --variants baseline
#
# Smoke test:
#   sbatch scripts/run_experiment.slurm --limit 50 --variants baseline,improved
#
# Any extra arguments are forwarded verbatim to run_experiment.py.

set -euo pipefail

# Activate the project conda environment.
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate llm-assignment

# NOTE: SLURM copies the batch script to a spool directory, so "$0" is NOT the
# path we were invoked from. SLURM_SUBMIT_DIR is the reliable way to get back to
# the directory the user ran `sbatch` in.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "host    : $(hostname)"
echo "cwd     : $PWD"
echo "job     : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "timelimit: ${SLURM_JOB_TIME_LIMIT:-?} on ${SLURM_JOB_PARTITION:-?}"
echo "start   : $(date '+%F %T')"
which python
nvidia-smi || true

export TOKENIZERS_PARALLELISM=false

python src/run_experiment.py "$@"

echo "end     : $(date '+%F %T')"
