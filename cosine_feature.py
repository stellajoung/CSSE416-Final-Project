"""Does adding the image-text cosine similarity as an input help, on frozen CLIP embeddings?

Compares, on the same rows and splits as the other experiments
  logreg concat          logistic regression on [image, text]
  logreg concat + cos    same, with the cosine similarity appended
  logreg cos only        logistic regression on the cosine similarity alone
  mlp concat             small MLP on [image, text]
  mlp concat + cos       same MLP, with the cosine similarity appended
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

C_GRID = [0.01, 0.1, 1, 10, 100]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/crisismmd_clean.csv")
    parser.add_argument("--emb", default="embeddings/crisismmd_text_clean.npz")
    parser.add_argument("--label-column", default="image_info")
    parser.add_argument("--all-rows", action="store_true", help="use disagreeing rows too")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="results/cosine_feature.csv")
    return parser.parse_args()


def load_data(csv_path, emb_path, all_rows):
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    data = np.load(emb_path)
    order = pd.Series(range(len(data["image_id"])), index=data["image_id"].astype(str))
    index = order.loc[frame["image_id"]].to_numpy()
    image_emb = data["image_emb"][index]
    text_emb = data["text_emb"][index]
    if not all_rows:
        keep = (frame["agreed"] == "True").to_numpy()
        frame, image_emb, text_emb = frame[keep].reset_index(drop=True), image_emb[keep], text_emb[keep]
    print(f"{len(frame)} rows, all_rows={all_rows}")
    return frame, image_emb, text_emb


def score(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_not_informative": per_class[0],
        "f1_informative": per_class[1],
    }


def zscore(x, train_mask):
    mean, std = x[train_mask].mean(0), x[train_mask].std(0) + 1e-6
    return (x - mean) / std


def run_logreg(name, x, y, masks):
    """Tune C on dev, train on train, report dev and test. Deterministic, so no seeds."""
    best = None
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        model.fit(x[masks["train"]], y[masks["train"]])
        dev_f1 = score(y[masks["dev"]], model.predict(x[masks["dev"]]))["macro_f1"]
        if best is None or dev_f1 > best[0]:
            best = (dev_f1, c, model)
    _, c, model = best
    print(f"  {name} chose C={c}")
    return [{"model": name, "seed": 0, "split": s, "setting": f"C={c}",
             **score(y[masks[s]], model.predict(x[masks[s]]))} for s in ["dev", "test"]]


def run_mlp(name, x, y, masks, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.tensor(zscore(x, masks["train"]), dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    idx = {s: torch.tensor(np.where(m)[0], device=device) for s, m in masks.items()}
    model = nn.Sequential(
        nn.Dropout(args.dropout), nn.Linear(x.shape[1], args.hidden), nn.ReLU(),
        nn.Dropout(args.dropout), nn.Linear(args.hidden, 2),
    ).to(device)
    counts = torch.bincount(yt[idx["train"]], minlength=2).float()
    loss_fn = nn.CrossEntropyLoss(weight=(len(idx["train"]) / (2 * counts)).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    def predict(split):
        model.eval()
        with torch.no_grad():
            logits = model(x[idx[split]])
            return loss_fn(logits, yt[idx[split]]).item(), logits.argmax(1).cpu().numpy()

    best = {"f1": -1.0, "state": None, "epoch": 0}
    n = len(idx["train"])
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = idx["train"][torch.randperm(n, device=device)]
        for start in range(0, n, args.batch_size):
            batch = order[start:start + args.batch_size]
            loss = loss_fn(model(x[batch]), yt[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dev_loss, dev_pred = predict("dev")
        dev_f1 = score(y[masks["dev"]], dev_pred)["macro_f1"]
        scheduler.step(dev_loss)
        if dev_f1 > best["f1"]:
            best = {"f1": dev_f1, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
        elif epoch - best["epoch"] >= args.patience:
            break
    model.load_state_dict(best["state"])
    print(f"  {name} seed {seed} best epoch {best['epoch']} dev macro F1 {best['f1']:.3f}")
    return [{"model": name, "seed": seed, "split": s, "setting": f"epoch={best['epoch']}",
             **score(y[masks[s]], predict(s)[1])} for s in ["dev", "test"]]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    frame, image_emb, text_emb = load_data(args.csv, args.emb, args.all_rows)
    y = (frame[args.label_column] == "informative").astype(int).to_numpy()
    masks = {s: (frame["split"] == s).to_numpy() for s in ["train", "dev", "test"]}
    print({s: int(m.sum()) for s, m in masks.items()}, f"informative share {y.mean():.3f}")

    cos = (image_emb * text_emb).sum(axis=1, keepdims=True)
    print(f"mean cosine, informative {cos[y == 1].mean():.3f}, not informative {cos[y == 0].mean():.3f}")
    concat = np.hstack([image_emb, text_emb])
    # the raw embeddings are tiny numbers, so z-score the cosine to give it a fair chance in the
    # logistic regression (the MLP z-scores every column itself)
    cos_z = zscore(cos, masks["train"])

    rows = []
    print("logistic regression")
    rows += run_logreg("logreg concat", concat, y, masks)
    rows += run_logreg("logreg concat + cos", np.hstack([concat, cos_z]), y, masks)
    rows += run_logreg("logreg cos only", cos_z, y, masks)
    print("mlp")
    for seed in range(args.seeds):
        rows += run_mlp("mlp concat", concat, y, masks, args, device, seed)
        rows += run_mlp("mlp concat + cos", np.hstack([concat, cos]), y, masks, args, device, seed)

    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    metrics = ["accuracy", "f1_not_informative", "f1_informative", "macro_f1"]
    for split in ["dev", "test"]:
        summary = table[table["split"] == split].groupby("model", sort=False)[metrics].agg(["mean", "std"])
        print(f"{split} results, mean and std over seeds (logistic regression has one run)")
        print(summary.round(3).to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
