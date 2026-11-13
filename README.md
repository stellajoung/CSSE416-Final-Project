# Multimodal Classification on CrisisMMD

Classifying disaster tweets (image + text) with CLIP. The main task is
**CrisisMMD Task 1: informative vs. not informative**. We compare frozen-CLIP
baselines, zero-shot CLIP, fusion heads and LoRA fine-tuned CLIP.

## Overview

The code has three stages:

1. **Data preparation.** `clean_dataset.py` reads the raw CrisisMMD annotations,
   cleans the tweet text, and writes one CSV (`data/crisismmd_clean.csv`) with
   image paths, labels and the train/dev/test split.
3. **Frozen-CLIP experiments.** `extract_embeddings.py` encodes every image and
   tweet once with CLIP and saves the vectors to `embeddings/`. Baselines,
   zero-shot and fusion-head experiments train on those saved vectors, so they are fast.
5. **Fine-tuning experiments.** CLIP is fine-tuned end to end with LoRA
   adapters on the top transformer layers. A small head fuses the image and text
   features. A variant fuses the token-level outputs with cross-attention.

Every experiment writes metrics (macro F1 and related scores) to `results/`.

## Installation

Python 3.9+ and a GPU are recommended. Fine-tuning and embedding extraction need
a GPU. The scripts are written for a Slurm cluster, but each one also runs as a
plain `python` command.

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install peft          # needed by the LoRA scripts
```
The CLIP weights download automatically from Hugging Face on first run. 

## Data

**Source:** CrisisMMD v2.0 (Alam, Ofli, Imran, "CrisisMMD: Multimodal Twitter
Datasets from Natural Disasters", ICWSM 2018). It has about 18,000 tweets and
images from seven 2017 disasters, annotated for three tasks. Download it from
the CrisisNLP resource page, <https://crisisnlp.qcri.org/crisismmd>, which hosts
the dataset and the official train/dev/test splits. The data is free for
research use and is not included in this repo.

**Where to put it:** make a `dataset/` folder in the repo root and extract the
downloads there.

```bash
mkdir -p dataset && cd dataset
tar -xzf CrisisMMD_v2.0.tar.gz
unzip crisismmd_datasplit_agreed_label.zip
cd ..
```

## How to run

Run the steps in this order. Each `.sbatch` file wraps the Python script of the
same name.

```bash
# 1. Clean the data
sbatch clean.sbatch

# 2. Extract frozen CLIP embeddings
sbatch extract.sbatch

# 3. Frozen-embedding experiments
sbatch baselines.sbatch
sbatch zeroshot.sbatch
sbatch cosine_feature.sbatch
sbatch cross_attention.sbatch

# 4. LoRA fine-tuning
sbatch CLIP_LoRA_3layers.sbatch        # LoRA on top layers, image+text
sbatch lora_token_fusion.sbatch        # token-level cross-attention fusion
```

## Code guide

| File | What it does |
|------|--------------|
| `clean_dataset.py` | Builds the cleaned CSV and splits |
| `extract_embeddings.py` | Frozen CLIP image and text embeddings |
| `baselines.py` | Logistic regression on image, text and concatenated embeddings |
| `zeroshot.py` | Zero-shot CLIP using prompt sets |
| `cosine_feature.py` | Tests whether image-text cosine similarity helps |
| `cross_attention.py` | Concat, self-attention and cross-attention fusion heads |
| `CLIP_Fine_Tuning_1.py` | Fine-tunes a layer-truncated CLIP fusion classifier |
| `CLIP_compare_runs.py` | Merges `metrics.json` files from fine-tuning runs into one table |
| `CLIP_LoRA_fusion_classifier.py` | LoRA on the top CLIP layers. Trains image, text and image+text models |
| `lora_token_fusion.py` | LoRA plus token-level cross-attention, compared with a pooled control |
| `zeroshot_CLIP_LoRA.py` | Zero-shot Tasks 2 and 3 using the LoRA-tuned encoders |
| `CLIP_LoRA_classifier_for_demo.py` | Loads a trained model and predicts for one image and text |
| `label_counts.py` | Prints label counts |

## Citation
```bibtex
@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec et al.},
  journal={ICML},
  year={2021}
}
```
