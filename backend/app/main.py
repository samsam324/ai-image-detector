from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import io

from .model import detector

app = FastAPI(title="AI Image Detector API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image_bytes(b: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    img = read_image_bytes(content)
    out = detector.predict(img)
    return out
