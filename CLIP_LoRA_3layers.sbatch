#!/bin/bash
#SBATCH --job-name=finetune_lora_3layers
#SBATCH --partition=gpu
#SBATCH --gres=shard:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/finetune_lora_3layers_job_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf_cache"

.venv/bin/python -u CLIP_LoRA_fusion_classifier.py \
    --modalities image+text \
    --lora-frac 0.25 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --out-dir results/lora_3layers \
    --image-root dataset/CrisisMMD_v2.0