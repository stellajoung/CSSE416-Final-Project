"""Zero-shot CrisisMMD Tasks 2 and 3 with the CLIP encoders from the Task 1 LoRA fusion
model (image+text run of finetune_clip_fusion.py), compared against frozen CLIP.
  Task 2 humanitarian  8 humanitarian categories
  Task 3 damage        severe / mild / little or no damage / don't know or can't judge

Method (zeroshot.py, extended to several prompts per class):
  - embeddings are read from an .npz (image_id, image_emb, text_emb), matched by image_id
  - each task has its own prompt-based classifier: every prompt is embedded and normalized,
    a class's prompts are averaged and renormalized into one class vector (CLIP prompt
    ensembling, Radford et al. 2021); a class with a single prompt is just that prompt
  - prompts are the hand-written lists HUMANITARIAN_SETS / DAMAGE_SETS, one set
    per task ("ensemble"); --prompts can supply several sets per task instead
  - image / text: argmax cosine against the class vectors
  - image+text: image sims + text sims, argmax
  - where a task has several prompt sets, the best per modality is chosen on dev macro F1,
    and test is reported

Embeddings per encoder (each .npz has both image_emb and text_emb):
  frozen  --emb, the .npz zeroshot.py used (embeddings/crisismmd_text_clean.npz)
  lora    --lora-emb, from the fusion model's adapter (--lora-adapter, both towers adapted).
          If the file is missing it is generated once over the whole CSV and saved, then
          reused on later runs.
All three zero-shot modalities (image, text, image+text) use the same encoder's embeddings.
Prompts are embedded by that encoder's text tower: base CLIP for frozen, the fusion
model's adapted text tower for lora.

"Don't know or can't judge" is kept as its own Task 3 class and needs prompts in
DAMAGE_SETS; pass --drop-cant-judge to remove it instead.

Outputs under --out-dir:
  zeroshot_other_tasks.csv            every task / encoder / modality / prompt set / split
  zeroshot_other_tasks_test_best.csv  best prompt set per task/encoder/modality, test split
  prompts.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

MODALITIES = ["image", "text", "image+text"]
CANT_JUDGE = re.compile(r"dont_know|don't_know|cant_judge|can't_judge|not_relevant_or_cant_judge")

# label column, and the column it must agree with (None = no agreement filter)
TASKS = {
    "humanitarian": {"label": "image_human", "agree": "text_human"},
    "damage": {"label": "image_damage", "agree": None},  # image-only annotation
}

# Prompts for Tasks 2/3: a list of prompts per class, averaged into one class vector.
# Edit here, or pass --prompts with {task: {set: {label: [prompts]}}} to replace a task's sets.
HUMANITARIAN_SETS = {
    "affected_individuals": [
        "a photo of disaster victims or displaced people needing aid",
        "a photo of civilians affected by a natural disaster sitting outside or in a shelter",
        "a photo showing people impacted by a hurricane or flood",
    ],
    "infrastructure_and_utility_damage": [
        "a photo of structural damage to buildings, roads, or bridges",
        "a photo of collapsed walls, flooded roads, or fallen power lines",
        "a photo of damaged infrastructure caused by a natural disaster",
    ],
    "injured_or_dead_people": [
        "a photo of injured people receiving medical emergency care during a disaster",
        "a photo of human casualties or injured individuals",
        "a photo showing wounded people or paramedics assisting victims",
    ],
    "missing_or_found_people": [
        "a photo or flyer of a missing person search notice",
        "a photo of a person missing or found after a natural disaster",
    ],
    "rescue_volunteering_or_donation_effort": [
        "a photo of first responders, rescue operations, or volunteer aid workers",
        "a photo of firefighters, coast guard rescue boats, or relief supply distributions",
        "a photo showing food and water donation drop-off locations",
    ],
    "vehicle_damage": [
        "a photo of damaged, submerged, or overturned vehicles and cars",
        "a photo of cars crushed by debris or flooded during a storm",
    ],
    "other_relevant_information": [
        "a photo or infographic showing disaster weather maps, updates, or news notices",
        "a photo of emergency announcement graphics or radar maps",
    ],
    "not_humanitarian": [ 
        "a photo of a normal everyday scene, selfie, meme, or unrelated topic",
        "a photo with no relation to a natural disaster or emergency response",
        "a clear photo of regular life with no crisis details",
    ],
}
DAMAGE_SETS = {
    "little_or_no_damage": [
        "a photo of intact buildings, usable roads, and undamaged infrastructure",
        "a photo showing minimal or no physical destruction after a disaster",
        "a photo of a normal building or road with no structural damage",
    ],
    "mild_damage": [
        "a photo showing partially damaged buildings, minor debris, or partially blocked roads",
        "a photo of moderate structural damage, loose tiles, or flooded streets with intact structures",
        "a photo of minor cracks, blown off shingles, or partial roof damage",
    ],
    "severe_damage": [
        "a photo of completely destroyed buildings, collapsed structures, or washed out roads",
        "a photo of total physical destruction, severe structural collapse, or unpassable sinkholes",
        "a photo showing heavy disaster destruction with leveled buildings or completely ruined bridges",
    ],
    "dont_know_or_cant_judge": [
        "a blurry, dark, or low-quality photo where any damage cannot be seen clearly",
        "a photo of text, a screenshot, or a graphic with no scene to assess for damage",
        "a close-up or ambiguous photo where the extent of the damage cannot be determined",
    ],
}
TASK_PROMPTS = {"humanitarian": HUMANITARIAN_SETS, "damage": DAMAGE_SETS}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/crisismmd_clean.csv")
    p.add_argument("--emb", default="embeddings/crisismmd_text_clean.npz", help="frozen CLIP embeddings")
    p.add_argument("--lora-adapter", default="results/lora/image_text/lora_adapter",
                   help="adapter from the image+text (fusion) fine-tuning run")
    p.add_argument("--lora-emb", default="embeddings/crisismmd_lora_fusion.npz",
                   help="fusion-model embeddings; generated here if missing")
    p.add_argument("--image-root", default=".", help="only used when generating LoRA embeddings")
    p.add_argument("--image-column", default="image_path")
    p.add_argument("--text-column", default="text_clean")
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--drop-cant-judge", action="store_true",
                   help="remove don't-know/can't-judge rows from Task 3 instead of scoring them")
    p.add_argument("--all-rows", action="store_true", help="skip the agreement filter")
    p.add_argument("--prompts", help="JSON {task: {set: {label: [prompts]}}} replacing that task's prompt sets")
    p.add_argument("--modalities", nargs="+", choices=MODALITIES, default=MODALITIES)
    p.add_argument("--encoders", nargs="+", choices=["frozen", "lora"], default=["frozen", "lora"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default="results/zeroshot_prompt_ensemble")
    return p.parse_args()


def uses_image(modality):
    return modality in ("image", "image+text")


def uses_text(modality):
    return modality in ("text", "image+text")


# ---------------------------------------------------------------- tasks and prompts

def setup_tasks(frame, args):
    overrides = json.loads(Path(args.prompts).read_text()) if args.prompts else {}
    tasks = []
    for name in args.tasks:
        label_col, agree_col = TASKS[name]["label"], TASKS[name]["agree"]
        if label_col not in frame.columns:
            raise SystemExit(f"task {name}: column '{label_col}' not in CSV. Columns: {list(frame.columns)}")

        # blank or "nan" labels mean the row has no annotation for this task (e.g. Task 3 only
        # labels some images); drop those rather than treating "nan" as a class
        missing_label = frame[label_col].str.strip().str.lower().isin(["", "nan", "none", "null"])
        rows = frame[frame["split"].isin(["dev", "test"]) & ~missing_label]
        cant_judge = [l for l in rows[label_col].unique() if CANT_JUDGE.search(l)]
        drop = args.drop_cant_judge or TASKS[name].get("drop_cant_judge", False)
        dropped = sorted(cant_judge) if drop else []
        rows = rows[~rows[label_col].isin(dropped)]
        if agree_col and not args.all_rows:
            if agree_col not in frame.columns:
                print(f"task {name}: no '{agree_col}' column, skipping agreement filter")
            else:
                rows = rows[rows[label_col] == rows[agree_col]]
        classes = sorted(rows[label_col].unique())

        if name in overrides:
            sets = overrides[name]
        else:
            sets = {"ensemble": TASK_PROMPTS[name]}
        # every class gets a list of prompts; only the classes being scored are kept
        sets = {s: {c: [p[c]] if isinstance(p.get(c), str) else list(p.get(c) or []) for c in classes}
                for s, p in sets.items()}
        for s, prompts in sets.items():
            missing = sorted(c for c in classes if not prompts[c])
            if missing:
                raise SystemExit(f"task {name}, prompt set {s}: no prompts for {missing}. Add them to the "
                                 f"task's prompt dict" + (" or run with --drop-cant-judge" if name == "damage" else ""))

        print(f"\ntask {name}: label={label_col} rows={len(rows)} classes={len(classes)} dropped={dropped}")
        print(rows.groupby([label_col, "split"]).size().unstack(fill_value=0).to_string())
        tasks.append({"name": name, "label_col": label_col, "classes": classes, "rows": rows, "sets": sets})
    return tasks


# ---------------------------------------------------------------- embeddings

class RowDataset(Dataset):
    def __init__(self, frame, args, modality):
        self.load_image = uses_image(modality)
        self.paths = [str(Path(args.image_root) / p) for p in frame[args.image_column]]
        self.texts = frame[args.text_column].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB") if self.load_image else None
        return image, self.texts[i]


def load_clip(args, encoder, device):
    clip = CLIPModel.from_pretrained(args.model)
    if encoder == "lora":
        path = Path(args.lora_adapter)
        if not path.exists():
            raise SystemExit(f"no adapter at {path}; run finetune_clip_fusion.py --modalities image+text first")
        clip = PeftModel.from_pretrained(clip, str(path)).merge_and_unload()
        print(f"loaded {path}")
    return clip.to(device).eval()


@torch.no_grad()
def export_lora_embeddings(clip, processor, frame, args, modality, device, path):
    """Embed every unique image_id in the CSV with the fusion model's encoders and save in
    the same .npz layout as the frozen embeddings."""
    frame = frame.drop_duplicates("image_id").reset_index(drop=True)
    print(f"generating {path} for {len(frame)} rows")
    # bf16 needs Ampere or newer (gus L40S); Turing cards (gebru RTX 6000) use fp16 instead
    amp = device == "cuda"
    amp_dtype = torch.bfloat16 if amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"autocast {amp_dtype if amp else 'off'} on {torch.cuda.get_device_name(0) if amp else 'cpu'}")

    def collate(batch):
        images, texts = zip(*batch)
        enc = {}
        if uses_image(modality):
            enc.update(processor(images=list(images), return_tensors="pt"))
        if uses_text(modality):
            enc.update(processor(text=list(texts), return_tensors="pt",
                                 padding=True, truncation=True, max_length=77))
        return enc

    loader = DataLoader(RowDataset(frame, args, modality), batch_size=args.batch_size,
                        num_workers=args.workers, collate_fn=collate, pin_memory=device == "cuda")
    img, txt = [], []
    for i, batch in enumerate(loader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp):
            if uses_image(modality):
                img.append(F.normalize(clip.get_image_features(
                    pixel_values=batch["pixel_values"]).float(), dim=-1).cpu())
            if uses_text(modality):
                txt.append(F.normalize(clip.get_text_features(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).float(), dim=-1).cpu())
        if i % 50 == 0:
            print(f"  batch {i}/{len(loader)}")
    out = {"image_id": frame["image_id"].to_numpy(dtype=str)}  # str, not object, so no pickle needed
    if img:
        out["image_emb"] = torch.cat(img).numpy()
    if txt:
        out["text_emb"] = torch.cat(txt).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def load_npz(path, frame):
    """Embeddings for the rows of `frame`, matched by image_id like zeroshot.py, L2-normalized
    (a no-op if they were already saved normalized)."""
    data = np.load(path, allow_pickle=True)  # older LoRA .npz files stored image_id as objects
    order = pd.Series(range(len(data["image_id"])), index=data["image_id"].astype(str))
    order = order[~order.index.duplicated()]
    missing = ~frame["image_id"].isin(order.index)
    if missing.any():
        raise SystemExit(f"{path}: {missing.sum()} rows have no embedding, e.g. {frame['image_id'][missing].iloc[0]}")
    index = order.loc[frame["image_id"]].to_numpy()
    out = {}
    for key in ["image_emb", "text_emb"]:
        if key in data.files:
            e = data[key][index].astype(np.float32)
            out[key] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return out


@torch.no_grad()
def encode_classes(clip, processor, class_prompts, device):
    """One vector per class: embed and normalize each prompt, average a class's prompts,
    renormalize. `class_prompts` is a list (one entry per class) of prompt lists."""
    flat = [p for prompts in class_prompts for p in prompts]
    inputs = processor(text=flat, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    emb = F.normalize(clip.get_text_features(**inputs).float(), dim=-1).cpu()
    out, start = [], 0
    for prompts in class_prompts:
        out.append(F.normalize(emb[start:start + len(prompts)].mean(0), dim=-1))
        start += len(prompts)
    return torch.stack(out).numpy()


# ---------------------------------------------------------------- scoring

def score(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        **{f"f1_{c}": f for c, f in zip(classes, per_class)},
    }


def save(rows, out_dir):
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "zeroshot_other_tasks.csv", index=False)
    # best prompt set per task/encoder/modality on dev, reported on test
    dev = table[table["split"] == "dev"]
    best = dev.loc[dev.groupby(["task", "encoder", "modality"])["macro_f1"].idxmax(),
                   ["task", "encoder", "modality", "prompt_set"]]
    chosen = table[table["split"] == "test"].merge(best, on=["task", "encoder", "modality", "prompt_set"])
    chosen = chosen.dropna(axis=1, how="all")
    chosen.to_csv(out_dir / "zeroshot_other_tasks_test_best.csv", index=False)
    return table, chosen


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    tasks = setup_tasks(frame, args)
    (out_dir / "prompts.json").write_text(json.dumps({t["name"]: t["sets"] for t in tasks}, indent=2))

    # every row any task needs; embeddings are looked up once per encoder, then sliced per task
    needed = frame.loc[sorted(set().union(*(t["rows"].index for t in tasks)))].reset_index()
    pos = pd.Series(np.arange(len(needed)), index=needed["index"])
    processor = CLIPProcessor.from_pretrained(args.model)

    rows = []
    for encoder in args.encoders:
        print(f"\n{'=' * 20} {encoder} {'=' * 20}")
        clip = load_clip(args, encoder, device)  # for prompts (and export if needed)
        if encoder == "frozen":
            emb_path = Path(args.emb)
        else:
            emb_path = Path(args.lora_emb)
            if not emb_path.exists():
                export_lora_embeddings(clip, processor, frame, args, "image+text", device, emb_path)
        print(f"embeddings {emb_path}")
        emb = load_npz(emb_path, needed)
        prompt_embs = {(t["name"], s): encode_classes(clip, processor, [p[c] for c in t["classes"]], device)
                       for t in tasks for s, p in t["sets"].items()}
        del clip
        if device == "cuda":
            torch.cuda.empty_cache()

        for modality in args.modalities:
            img = emb["image_emb"] if uses_image(modality) else None
            txt = emb["text_emb"] if uses_text(modality) else None

            for task in tasks:
                classes = task["classes"]
                for set_name in task["sets"]:
                    prompt_emb = prompt_embs[(task["name"], set_name)]
                    sim = 0
                    if img is not None:
                        sim = sim + img @ prompt_emb.T
                    if txt is not None:
                        sim = sim + txt @ prompt_emb.T
                    pred_all = sim.argmax(axis=1)

                    for split in ["dev", "test"]:
                        part = task["rows"][task["rows"]["split"] == split]
                        if part.empty:
                            continue
                        y = part[task["label_col"]].map({c: i for i, c in enumerate(classes)}).to_numpy()
                        pred = pred_all[pos[part.index].to_numpy()]
                        rows.append({"task": task["name"], "encoder": encoder, "prompt_set": set_name,
                                     "modality": modality, "split": split, **score(y, pred, classes)})

        save(rows, out_dir)  # after every encoder, so a crash doesn't lose finished runs

    table, chosen = save(rows, out_dir)
    print("\ndev macro F1 by task / modality / prompt set")
    dev = table[table["split"] == "dev"]
    print(dev.pivot_table(index=["task", "modality", "prompt_set"], columns="encoder",
                          values="macro_f1").round(3).to_string())
    print("\ntest, best prompt set per task/encoder/modality (chosen on dev)")
    cols = ["task", "encoder", "modality", "prompt_set", "accuracy", "macro_f1", "weighted_f1"]
    print(chosen[cols].round(3).to_string(index=False))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()