"""LoRA fine-tuned CLIP with token-level cross-attention fusion on CrisisMMD Task 1.

The bottom CLIP layers stay frozen. LoRA adapters go on the attention projections of the top
N layers of both the image and the text encoder. The fusion head works on the encoders' token
outputs instead of one pooled vector per modality.

--fusion token   Image patch tokens and text tokens are projected to a shared width and passed
                 through cross-attention blocks. In each block the image tokens attend to the
                 text tokens and the text tokens attend to the image tokens. Each stream is
                 mean-pooled and the result goes to an MLP classifier. By default the pooled CLIP
                 embeddings [img, txt, img*txt] are also fed to the classifier (skip connection),
                 so any gain over "pooled" comes from the token branch. Use --no-skip to drop it.
--fusion pooled  Control. Only [img, txt, img*txt] of the pooled CLIP embeddings goes to the same
                 MLP classifier, with no token branch.

Everything else (LoRA, data, optimizer, schedule, seeds) is identical between the two.

To put a fusion head on an already trained LoRA model, pass --adapter-dir (the saved adapter folder)
with the same --lora-layers and --lora-r it was trained with, and --freeze-lora to train only the head.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/crisismmd_clean.csv")
    p.add_argument("--image-root", default="dataset/CrisisMMD_v2.0", help="prefix joined to image_path")
    p.add_argument("--image-column", default="image_path")
    p.add_argument("--text-column", default="text_clean")
    p.add_argument("--label-column", default="image_info")
    p.add_argument("--all-rows", action="store_true", help="use disagreeing rows too")
    p.add_argument("--limit", type=int, default=0, help="random sample of rows for a quick smoke test")
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--fusions", nargs="+", choices=["token", "pooled"], default=["token", "pooled"])
    # LoRA
    p.add_argument("--lora-layers", type=int, default=4, help="top N layers of each encoder get LoRA")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--adapter-dir", default=None,
                   help="load trained LoRA adapter weights (a PEFT save_pretrained folder) before training")
    p.add_argument("--freeze-lora", action="store_true",
                   help="do not train the LoRA weights, only the fusion head and classifier")
    # token fusion head
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--fusion-layers", type=int, default=2)
    p.add_argument("--fusion-dropout", type=float, default=0.1)
    p.add_argument("--no-skip", action="store_true", help="token fusion without the pooled-embedding skip")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2, help="dropout in the classifier MLP")
    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="warmup fraction of total steps")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out-dir", default="results/lora_token")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data

def load_frame(args):
    frame = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    if not args.all_rows:
        frame = frame[frame["agreed"] == "True"].reset_index(drop=True)
    if args.limit:
        frame = frame.sample(n=min(args.limit, len(frame)), random_state=0).reset_index(drop=True)
    frame["y"] = (frame[args.label_column] == "informative").astype(int)
    print(f"{len(frame)} rows, all_rows={args.all_rows}, informative share {frame['y'].mean():.3f}")
    print(frame.groupby("split")["y"].agg(["count", "mean"]).round(3).to_string())
    return frame


class CrisisDataset(Dataset):
    def __init__(self, frame, args):
        self.paths = [str(Path(args.image_root) / p) for p in frame[args.image_column]]
        self.texts = frame[args.text_column].tolist()
        self.labels = frame["y"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert("RGB"), self.texts[i], self.labels[i]


def make_collate(processor):
    def collate(batch):
        images, texts, labels = zip(*batch)
        enc = processor(images=list(images), text=list(texts), return_tensors="pt",
                        padding=True, truncation=True, max_length=77)
        return {"pixel_values": enc["pixel_values"], "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": torch.tensor(labels, dtype=torch.long)}
    return collate


# ---------------------------------------------------------------- model

def lora_target_regex(clip, layers):
    """Match q/k/v/out projections in the top `layers` layers of both encoders."""
    parts = []
    for tower, cfg in [("vision_model", clip.config.vision_config), ("text_model", clip.config.text_config)]:
        n = cfg.num_hidden_layers
        k = min(layers, n)
        top = list(range(n - k, n))
        print(f"{tower}: {n} layers, frozen 0-{n - k - 1}, LoRA on {top}")
        idx = "|".join(map(str, top))
        parts.append(rf"{tower}\.encoder\.layers\.({idx})\.self_attn\.(q_proj|k_proj|v_proj|out_proj)")
    return r".*(" + "|".join(parts) + r")"


class CrossBlock(nn.Module):
    """Image tokens attend to text tokens and text tokens attend to image tokens, then a
    feed-forward layer on each stream. Both directions use the inputs of the block."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn_img = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_txt = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_img1, self.norm_img2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm_txt1, self.norm_txt2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff_img = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.ff_txt = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, img, txt, txt_pad):
        img_update, _ = self.attn_img(img, txt, txt, key_padding_mask=txt_pad)
        txt_update, _ = self.attn_txt(txt, img, img)
        img = self.norm_img1(img + self.drop(img_update))
        txt = self.norm_txt1(txt + self.drop(txt_update))
        img = self.norm_img2(img + self.drop(self.ff_img(img)))
        txt = self.norm_txt2(txt + self.drop(self.ff_txt(txt)))
        return img, txt


def masked_mean(x, keep):
    keep = keep.unsqueeze(-1).to(x.dtype)
    return (x * keep).sum(1) / keep.sum(1).clamp(min=1)


class Classifier(nn.Module):
    def __init__(self, clip, fusion, dims, args):
        super().__init__()
        self.clip = clip
        self.fusion = fusion
        self.skip = not args.no_skip
        proj = dims["proj"]
        in_dim = 3 * proj
        if fusion == "token":
            d = args.d_model
            self.proj_img = nn.Sequential(nn.Linear(dims["vision"], d), nn.LayerNorm(d))
            self.proj_txt = nn.Sequential(nn.Linear(dims["text"], d), nn.LayerNorm(d))
            self.blocks = nn.ModuleList(
                [CrossBlock(d, args.heads, args.fusion_dropout) for _ in range(args.fusion_layers)])
            in_dim = 2 * d + (3 * proj if self.skip else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, args.hidden), nn.GELU(), nn.Dropout(args.dropout),
            nn.Linear(args.hidden, 2),
        )

    def forward(self, batch):
        vision = self.clip.vision_model(pixel_values=batch["pixel_values"])
        text = self.clip.text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        img_emb = F.normalize(self.clip.visual_projection(vision.pooler_output), dim=-1)
        txt_emb = F.normalize(self.clip.text_projection(text.pooler_output), dim=-1)
        pooled = torch.cat([img_emb, txt_emb, img_emb * txt_emb], dim=-1)
        if self.fusion == "pooled":
            return self.head(pooled)

        keep = batch["attention_mask"].bool()
        img = self.proj_img(self.clip.vision_model.post_layernorm(vision.last_hidden_state))
        txt = self.proj_txt(text.last_hidden_state)
        for block in self.blocks:
            img, txt = block(img, txt, ~keep)
        features = [img.mean(1), masked_mean(txt, keep)]
        if self.skip:
            features.append(pooled)
        return self.head(torch.cat(features, dim=-1))


def build_model(args, fusion, device):
    # fresh pretrained CLIP for every run, so runs never share weights
    clip = CLIPModel.from_pretrained(args.model)
    dims = {"vision": clip.config.vision_config.hidden_size,
            "text": clip.config.text_config.hidden_size,
            "proj": clip.config.projection_dim}
    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        target_modules=lora_target_regex(clip, args.lora_layers), bias="none")
    # get_peft_model freezes every CLIP weight, only the LoRA matrices train
    clip = get_peft_model(clip, config)
    if args.adapter_dir:
        result = set_peft_model_state_dict(clip, load_peft_weights(args.adapter_dir, device="cpu"))
        unexpected = list(getattr(result, "unexpected_keys", []))
        if unexpected:
            raise SystemExit(f"adapter does not match this LoRA setup (check --lora-layers and --lora-r), "
                             f"{len(unexpected)} unexpected keys, first {unexpected[0]}")
        print(f"loaded LoRA adapter from {args.adapter_dir}")
    if args.freeze_lora:
        for p in clip.parameters():
            p.requires_grad = False
    clip.print_trainable_parameters()
    model = Classifier(clip, fusion, dims, args).to(device)
    n_head = sum(p.numel() for p in model.parameters() if p.requires_grad) - \
        sum(p.numel() for p in model.clip.parameters() if p.requires_grad)
    print(f"fusion head and classifier parameters {n_head:,}")
    return model


# ---------------------------------------------------------------- train / eval

def score(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_not_informative": per_class[0],
        "f1_informative": per_class[1],
    }


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
            logits = model(batch)
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
        ys.append(batch["labels"].cpu())
    prob, y = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    pred = (prob > 0.5).astype(int)
    return score(y, pred), y, pred, prob


def run_one(args, fusion, seed, frame, processor, device, amp, out_root):
    print(f"\n{'=' * 20} fusion {fusion}, seed {seed} {'=' * 20}")
    seed_everything(seed)
    parts = {s: frame[frame["split"] == s].reset_index(drop=True) for s in ["train", "dev", "test"]}
    loaders = {s: DataLoader(CrisisDataset(part, args), batch_size=args.batch_size, shuffle=s == "train",
                             num_workers=args.workers, collate_fn=make_collate(processor),
                             pin_memory=device == "cuda") for s, part in parts.items()}

    model = build_model(args, fusion, device)
    lora_params = [p for p in model.clip.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("clip.")]
    groups = [{"params": head_params, "lr": args.lr_head}]
    if lora_params:
        groups.insert(0, {"params": lora_params, "lr": args.lr_lora})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total = args.epochs * len(loaders["train"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.warmup * total), total)

    weight = None
    if not args.no_class_weights:
        counts = np.bincount(parts["train"]["y"], minlength=2)
        weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float, device=device)

    run_dir = out_root / f"{fusion}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_lora:
            model.clip.eval()  # frozen encoders, so no LoRA dropout noise
        running = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch)
            loss = F.cross_entropy(logits.float(), batch["labels"], weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"[{fusion}] epoch {epoch} step {step}/{len(loaders['train'])} loss {running / step:.4f}",
                      flush=True)
        dev, _, _, _ = evaluate(model, loaders["dev"], device, amp)
        history.append({"epoch": epoch, "train_loss": running / len(loaders["train"]),
                        **{f"dev_{k}": v for k, v in dev.items()}})
        print(f"[{fusion}] epoch {epoch} dev " + " ".join(f"{k}={v:.3f}" for k, v in dev.items()), flush=True)
        if dev["macro_f1"] > best["f1"]:
            best = {"f1": dev["macro_f1"], "epoch": epoch,
                    "state": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}}
            print(f"  new best dev macro F1 {best['f1']:.3f}")

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best["state"]:
                p.copy_(best["state"][n].to(p.device))
    torch.save(best["state"], run_dir / "best_trainable.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    rows = []
    for split in ["dev", "test"]:
        metrics, y, pred, prob = evaluate(model, loaders[split], device, amp)
        rows.append({"fusion": fusion, "seed": seed, "split": split, "best_epoch": best["epoch"], **metrics})
        if split == "test":
            out = parts["test"][["image_id", "event"]].copy()
            out["label"], out["pred"], out["prob_informative"] = y, pred, prob
            out.to_csv(run_dir / "test_preds.csv", index=False)
    print(f"[{fusion}] best epoch {best['epoch']}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    del model, optimizer, scheduler
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # real bf16 needs an Ampere or newer GPU, older cards fall back to full precision
    amp = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    print(f"device {device}, bf16 autocast {amp}")

    frame = load_frame(args)
    processor = CLIPProcessor.from_pretrained(args.model)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "args.json").write_text(json.dumps(vars(args), indent=2))

    rows = []
    for fusion in args.fusions:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            rows += run_one(args, fusion, seed, frame, processor, device, amp, out_root)
            # write after every run so a crash later does not lose finished runs
            pd.DataFrame(rows).to_csv(out_root / "results.csv", index=False)

    table = pd.DataFrame(rows)
    metrics = ["accuracy", "f1_not_informative", "f1_informative", "macro_f1"]
    for split in ["dev", "test"]:
        summary = table[table["split"] == split].groupby("fusion", sort=False)[metrics].agg(["mean", "std"])
        print(f"\n{split} results, mean and std over seeds")
        print(summary.round(3).to_string())
    print(f"saved {out_root}")


if __name__ == "__main__":
    main()