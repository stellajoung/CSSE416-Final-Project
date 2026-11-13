"""Inference for the CLIP + LoRA fusion classifier trained by CLIP_LoRA_fusion_classifier.py
(CrisisMMD Task 1: informative vs not informative).

Usage from a backend:

    from crisis_classifier import load_model, predict

    model = load_model("results/lora_3layers/image_text")      # once, at startup
    p_inf, p_not = predict(model, image, text)                   # per request

`image` can be a file path, raw bytes (e.g. an uploaded file's contents), a file-like
object, or a PIL image. `text` is the tweet/caption that goes with the image; the image+text
model needs it. Probabilities sum to 1.

What load_model rebuilds, exactly as in training:
  - base CLIP (openai/clip-vit-base-patch32 unless the run used another --model)
  - the LoRA adapter from <model_dir>/lora_adapter, merged into the CLIP weights
  - the MLP head from <model_dir>/head.pt: Linear -> GELU -> Dropout -> Linear (2 classes)
  - features: normalized image and text embeddings; for image+text, [img, txt, img * txt]
Class order matches training: index 0 = not_informative, index 1 = informative.

The saved .npz embedding files are not needed: they only cover the CrisisMMD dataset,
while uploaded images are embedded on the fly.

Quick test from the command line:
    python crisis_classifier.py --model-dir results/lora_3layers/image_text \\
        --image some_photo.jpg --text "Flooding on Main St, families need rescue"
"""
import argparse
import io
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

LABELS = ["not_informative", "informative"]  # same order as training
DEFAULT_BASE_MODEL = "openai/clip-vit-base-patch32"


class CrisisClassifier(nn.Module):
    """CLIP encoders (LoRA merged in) + the trained MLP head."""

    def __init__(self, clip, head, processor, modality, device):
        super().__init__()
        self.clip = clip
        self.head = head
        self.processor = processor
        self.modality = modality
        self.device = device

    @property
    def uses_image(self):
        return self.modality in ("image", "image+text")

    @property
    def uses_text(self):
        return self.modality in ("text", "image+text")

    @torch.no_grad()
    def forward_probs(self, images, texts):
        """images: list of PIL images (or None); texts: list of str (or None).
        Returns a (batch, 2) tensor of probabilities in LABELS order."""
        if self.uses_image:
            pix = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            img = F.normalize(self.clip.get_image_features(pixel_values=pix), dim=-1)
        if self.uses_text:
            enc = self.processor(text=texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=77).to(self.device)
            txt = F.normalize(self.clip.get_text_features(input_ids=enc["input_ids"],
                                                          attention_mask=enc["attention_mask"]), dim=-1)
        if self.modality == "image":
            features = img
        elif self.modality == "text":
            features = txt
        else:
            features = torch.cat([img, txt, img * txt], dim=-1)
        return self.head(features).float().softmax(dim=-1)


def _modality_from_dir(model_dir):
    name = Path(model_dir).name
    return {"image_text": "image+text", "image": "image", "text": "text"}.get(name)


def _base_model_name(model_dir):
    """The --model used in training, from args.json one level up, if it is there."""
    args_file = Path(model_dir).parent / "args.json"
    if args_file.exists():
        try:
            return json.loads(args_file.read_text()).get("model", DEFAULT_BASE_MODEL)
        except (OSError, ValueError):
            pass
    return DEFAULT_BASE_MODEL


def load_model(model_dir="results/lora_3layers/image_text", modality=None, base_model=None, device=None):
    """Load base CLIP, merge the LoRA adapter, load the head. Call once and reuse.

    model_dir   folder with lora_adapter/ and head.pt (one modality's --out-dir subfolder)
    modality    "image+text", "image" or "text"; inferred from the folder name if omitted
    base_model  Hugging Face CLIP name; read from the run's args.json if omitted
    device      "cuda" or "cpu"; picks cuda when available if omitted
    """
    model_dir = Path(model_dir)
    adapter_dir, head_path = model_dir / "lora_adapter", model_dir / "head.pt"
    for p in (adapter_dir, head_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; is {model_dir} a trained model folder?")
    modality = modality or _modality_from_dir(model_dir)
    if modality not in ("image+text", "image", "text"):
        raise ValueError(f"can't tell the modality from '{model_dir.name}'; pass modality=")
    base_model = base_model or _base_model_name(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    clip = CLIPModel.from_pretrained(base_model)
    clip = PeftModel.from_pretrained(clip, str(adapter_dir)).merge_and_unload()
    processor = CLIPProcessor.from_pretrained(base_model)

    # same head as training: Linear(in, 512) -> GELU -> Dropout(0.2) -> Linear(512, 2)
    state = torch.load(head_path, map_location="cpu")
    hidden, in_dim = state["0.weight"].shape
    n_classes = state["3.weight"].shape[0]
    head = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, n_classes))
    head.load_state_dict(state)

    expected = 3 * clip.config.projection_dim if modality == "image+text" else clip.config.projection_dim
    if in_dim != expected:
        raise ValueError(f"head expects {in_dim} features but modality '{modality}' gives {expected}; "
                         f"wrong modality or folder?")

    model = CrisisClassifier(clip, head, processor, modality, device).to(device).eval()
    return model


def _to_pil(image):
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif hasattr(image, "read"):  # file-like, e.g. an upload stream
        pil = Image.open(image)
    else:
        raise TypeError(f"unsupported image type {type(image).__name__}")
    return pil.convert("RGB")


def predict(model, image=None, text=None):
    """Classify one input. Returns (prob_informative, prob_not_informative)."""
    if model.uses_image and image is None:
        raise ValueError("this model needs an image")
    if model.uses_text and (text is None or not str(text).strip()):
        raise ValueError("this model needs the tweet/caption text that goes with the image")
    images = [_to_pil(image)] if model.uses_image else None
    texts = [str(text)] if model.uses_text else None
    probs = model.forward_probs(images, texts)[0].cpu().tolist()
    return probs[LABELS.index("informative")], probs[LABELS.index("not_informative")]


def predict_batch(model, images=None, texts=None):
    """Classify several inputs at once. Returns a list of (prob_informative, prob_not_informative)."""
    n = len(images) if images is not None else len(texts)
    pil = [_to_pil(im) for im in images] if model.uses_image else None
    txt = [str(t) for t in texts] if model.uses_text else None
    probs = model.forward_probs(pil, txt).cpu().tolist()
    i_inf, i_not = LABELS.index("informative"), LABELS.index("not_informative")
    return [(p[i_inf], p[i_not]) for p in probs][:n]


inference = predict  # alias, if the frontend expects inference()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Classify one image (+ text) as informative or not")
    ap.add_argument("--model-dir", default="results/lora_3layers/image_text")
    ap.add_argument("--image")
    ap.add_argument("--text")
    a = ap.parse_args()
    m = load_model(a.model_dir)
    p_inf, p_not = predict(m, a.image, a.text)
    print(f"informative: {p_inf:.4f}   not informative: {p_not:.4f}")