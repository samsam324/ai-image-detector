from __future__ import annotations

import argparse
import json
import os
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


def iter_images(root: str) -> Iterator[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(dirpath, f)


def label_from_path(path: str) -> int:
    parts = path.replace("\\", "/").split("/")
    if "real" in parts:
        return 0
    if "ai" in parts:
        return 1
    raise ValueError(f"Path must include /real/ or /ai/: {path}")


def load_img(path: str, max_w: int = 800) -> Image.Image:
    # Some files may be corrupted / mislabeled; caller should handle exceptions.
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    # Force decode now so we catch truncation/corruption inside try/except
    img.load()

    if img.width > max_w:
        s = max_w / img.width
        img = img.resize((int(img.width * s), int(img.height * s)))
    return img


def _get_clip(device: str, clip_model_name: str, clip_pretrained: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained, device=device
    )
    model.eval()
    return model, preprocess


def _infer_embed_dim(model, device: str) -> int:
    # OpenCLIP ViT-B-32 embed dim is usually 512, but infer to be safe.
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        d = model.encode_image(dummy).shape[-1]
    return int(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--max-w", type=int, default=800, help="Resize images if width exceeds this.")
    ap.add_argument("--log-every", type=int, default=1000, help="Progress logging cadence for extra info.")
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    data_root = os.path.join(project_root, "data", args.split)

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)

    clip_model_name = "ViT-B-32"
    clip_pretrained = "laion2b_s34b_b79k"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model, preprocess = _get_clip(device, clip_model_name, clip_pretrained)
    d = _infer_embed_dim(model, device)

    paths = list(iter_images(data_root))
    if not paths:
        raise SystemExit(
            f"No images found under {data_root} (expected .../{args.split}/real and .../{args.split}/ai)"
        )

    # We write into a temp memmap sized for all paths, but only the first `kept`
    # rows are valid (we skip corrupted/unreadable images). At the end, we save
    # compact .npy arrays with only valid rows.
    tmp_X_path = os.path.join(out_dir, f"X_{args.split}.tmp.npy")
    tmp_y_path = os.path.join(out_dir, f"y_{args.split}.tmp.npy")

    X_tmp = np.lib.format.open_memmap(
        tmp_X_path, mode="w+", dtype=np.float32, shape=(len(paths), d)
    )
    y_tmp = np.lib.format.open_memmap(
        tmp_y_path, mode="w+", dtype=np.int64, shape=(len(paths),)
    )

    rows = []
    bad_log = os.path.join(out_dir, f"bad_{args.split}.txt")
    kept = 0
    bad = 0

    pbar = tqdm(paths, desc=f"Embedding {args.split}")
    for idx, p in enumerate(pbar):
        try:
            img = load_img(p, max_w=args.max_w)
            t = preprocess(img).unsqueeze(0).to(device)

            with torch.inference_mode():
                feat = model.encode_image(t)
                feat = feat / feat.norm(dim=-1, keepdim=True)

            emb = feat.detach().cpu().numpy().astype(np.float32)[0]
            lbl = label_from_path(p)

            X_tmp[kept] = emb
            y_tmp[kept] = lbl
            rows.append({"path": p, "label": lbl})
            kept += 1

        except (UnidentifiedImageError, OSError, ValueError) as e:
            bad += 1
            with open(bad_log, "a", encoding="utf-8") as f:
                f.write(f"{p}\t{type(e).__name__}: {e}\n")
            continue

        if args.log_every and (idx + 1) % args.log_every == 0:
            pbar.set_postfix(kept=kept, bad=bad)

    # Flush memmaps
    del X_tmp
    del y_tmp

    # Load temp arrays via mmap and save compact outputs (only valid rows)
    X_tmp = np.load(tmp_X_path, mmap_mode="r")
    y_tmp = np.load(tmp_y_path, mmap_mode="r")

    X_final_path = os.path.join(out_dir, f"X_{args.split}.npy")
    y_final_path = os.path.join(out_dir, f"y_{args.split}.npy")
    man_path = os.path.join(out_dir, f"manifest_{args.split}.csv")

    # Materialize only `kept` rows to a compact file
    np.save(X_final_path, np.asarray(X_tmp[:kept], dtype=np.float32))
    np.save(y_final_path, np.asarray(y_tmp[:kept], dtype=np.int64))
    pd.DataFrame(rows).to_csv(man_path, index=False)

    # Remove temp files (optional but keeps folder clean)
    try:
        os.remove(tmp_X_path)
    except OSError:
        pass
    try:
        os.remove(tmp_y_path)
    except OSError:
        pass

    # Save clip config for backend
    artifact_dir = os.path.join(project_root, "backend", "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "clip_config.json"), "w", encoding="utf-8") as f:
        json.dump({"clip_model_name": clip_model_name, "clip_pretrained": clip_pretrained}, f, indent=2)

    print(
        f"Saved {args.split} embeddings to {out_dir}:\n"
        f"  {os.path.basename(X_final_path)}\n"
        f"  {os.path.basename(y_final_path)}\n"
        f"  {os.path.basename(man_path)}\n"
        f"kept={kept} bad={bad} (bad log: {bad_log if bad else 'none'})"
    )


if __name__ == "__main__":
    main()
