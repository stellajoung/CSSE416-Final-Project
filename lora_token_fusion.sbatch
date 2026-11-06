#!/bin/bash
#SBATCH --job-name=lora-token
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf_cache"
nvidia-smi

# peft is needed for LoRA and is not in requirements.txt
.venv/bin/python -c "import peft" 2>/dev/null || .venv/bin/pip install -q peft

.venv/bin/python lora_token_fusion.py --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
