from __future__ import annotations
from PIL import Image, ImageOps
import numpy as np

def load_image(file_bytes: bytes, max_w: int = 1400) -> Image.Image:
    img = Image.open(io_bytes(file_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    if img.width > max_w:
        s = max_w / img.width
        img = img.resize((int(img.width * s), int(img.height * s)))
    return img

def io_bytes(b: bytes):
    import io
    return io.BytesIO(b)

def pil_to_np(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr
