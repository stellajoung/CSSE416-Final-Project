#!/bin/bash
#SBATCH --job-name=zeroshot_CLIP_LoRA
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/zeroshot_CLIP_LoRA_job_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf_cache"

.venv/bin/python zeroshot_CLIP_LoRA.py "$@"
