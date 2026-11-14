#!/bin/bash
# GPU stage "judge" on DCC. See scripts/run_stage.sh for what it runs and its env-var options.
#   sbatch legacy/slurm/03_judge.sbatch
#SBATCH --job-name=tb-judge
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source legacy/slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
PROFILE=none bash scripts/run_stage.sh judge
