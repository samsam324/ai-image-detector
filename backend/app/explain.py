from __future__ import annotations
import numpy as np
from PIL import Image

def explanation_overlay(img: Image.Image, alpha: float = 0.40) -> Image.Image:
    base = img.convert("RGB")
    arr = np.asarray(base).astype(np.float32) / 255.0
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])

    gx, gy = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-8)
    mag = np.clip(mag**0.85, 0, 1)

    heat = np.zeros_like(arr)
    heat[..., 0] = mag
    heat[..., 1] = mag * 0.10
    heat[..., 2] = (1.0 - mag) * 0.20

    out = (1 - alpha) * arr + alpha * heat
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))

def png_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
