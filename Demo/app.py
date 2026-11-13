# app.py
import CLIP_LoRA_classifier_for_demo as CLIP

import io
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bonillml.github.io"],  # no repo path, no trailing slash
    allow_methods=["*"],
    allow_headers=["*"],
)

#device = "mps" if torch.backends.mps.is_available() else (
#    "cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"
model = CLIP.load_model()            # your code: build model + load weights
model.to(device).eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    mode: str = Form(...),
    text: str = Form(""),
    images: list[UploadFile] = File(default=[]),
):
    if mode != "pair":
        raise HTTPException(400, "Only Image + text mode is available in this demo. Switch to that tab.")
    if not text.strip():
        raise HTTPException(400, "Please enter the tweet text.")
    if not images:
        raise HTTPException(400, "Please add at least one image.")

    per_image = []
    for f in images[:4]:
        try:
            img = Image.open(io.BytesIO(await f.read())).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Couldn't read {f.filename} as an image.")
        with torch.no_grad():
            # Your inference code. Must return [p_not_informative, p_informative]
            p_inf, p_not = CLIP.inference(model, img, text)
        per_image.append({"informative": float(p_inf), "not_informative": float(p_not)})

    # Combine multiple images by averaging
    avg_inf = sum(p["informative"] for p in per_image) / len(per_image)

    return {
        "probabilities": {"informative": avg_inf, "not_informative": 1 - avg_inf},
        "per_image": [{"probabilities": p} for p in per_image],
    }