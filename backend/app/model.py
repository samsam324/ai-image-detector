from __future__ import annotations
import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

import joblib

import torch
import open_clip

from .explain import explanation_overlay, png_bytes


def _soft_clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def confidence_label(p: float) -> str:
    d = abs(p - 0.5) * 2.0
    if d >= 0.70:
        return "High"
    if d >= 0.42:
        return "Medium"
    return "Low"


def _demo_heuristic_probability(img: Image.Image) -> Tuple[float, Dict[str, float]]:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])

    gx, gy = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-8)

    edge_signal = float(np.clip(np.mean(mag**1.2) / 0.18, 0, 1))
    texture_signal = float(np.clip(np.std(gray) / 0.22, 0, 1))

    raw = 0.55 * edge_signal + 0.45 * (1.0 - texture_signal)

    prob = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.45)))
    prob = float(np.clip(prob, 0.05, 0.95))

    details = {
        "Edge artifacts": edge_signal,
        "Texture consistency": 1.0 - texture_signal,
    }
    return prob, details


@dataclass
class LoadedModel:
    clip_model_name: str
    clip_pretrained: str
    device: str
    model: Any
    preprocess: Any
    clf: Any
    calibrator: Optional[Any]  # may be None


class DetectorBackend:
    def __init__(self) -> None:
        self.loaded: Optional[LoadedModel] = None
        self._load_if_available()

    def _artifact_path(self, name: str) -> str:
        return os.path.join(ARTIFACT_DIR, name)

    def _load_if_available(self) -> None:
        cfg_path = self._artifact_path("clip_config.json")
        clf_path = self._artifact_path("clf.joblib")
        cal_path = self._artifact_path("calibrator.joblib")

        if not (os.path.exists(cfg_path) and os.path.exists(clf_path)):
            self.loaded = None
            print("[backend] No trained artifacts found. Running in DEMO fallback mode.")
            print(f"[backend] Expected at least: {cfg_path} and {clf_path}")
            return

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["clip_model_name"], pretrained=cfg["clip_pretrained"], device=device
        )
        model.eval()

        clf = joblib.load(clf_path)
        calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None

        self.loaded = LoadedModel(
            clip_model_name=cfg["clip_model_name"],
            clip_pretrained=cfg["clip_pretrained"],
            device=device,
            model=model,
            preprocess=preprocess,
            clf=clf,
            calibrator=calibrator,
        )
        print("[backend] Loaded trained artifacts successfully.")
        print(f"[backend] Device: {device} | CLIP: {cfg['clip_model_name']} ({cfg['clip_pretrained']})")

    @torch.inference_mode()
    def _embed_clip(self, img: Image.Image) -> np.ndarray:
        assert self.loaded is not None
        t = self.loaded.preprocess(img).unsqueeze(0).to(self.loaded.device)
        feats = self.loaded.model.encode_image(t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)[0]

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        overlay = explanation_overlay(img, alpha=0.40)
        overlay_b64 = base64.b64encode(png_bytes(overlay)).decode("utf-8")

        if self.loaded is None:
            prob, signals = _demo_heuristic_probability(img)
            verdict = "likely_ai" if prob >= 0.60 else "likely_real"
            return {
                "ai_probability": prob,
                "verdict": verdict,
                "confidence": confidence_label(prob),
                "signals": signals,
                "explanation_map_png_base64": overlay_b64,
                "mode": "demo_fallback",
            }

        emb = self._embed_clip(img)
        if hasattr(self.loaded.clf, "predict_proba"):
            p = float(self.loaded.clf.predict_proba([emb])[0][1])
        else:
            s = float(self.loaded.clf.decision_function([emb])[0])
            p = float(1.0 / (1.0 + np.exp(-s)))

        if self.loaded.calibrator is not None:
            p = float(self.loaded.calibrator.predict_proba([emb])[0][1])

        p = _soft_clip(p, 0.01, 0.99)
        verdict = "likely_ai" if p >= 0.60 else "likely_real"

        signals = {
            "CLIP embedding score": float(_soft_clip(p, 0.0, 1.0)),
        }

        return {
            "ai_probability": p,
            "verdict": verdict,
            "confidence": confidence_label(p),
            "signals": signals,
            "explanation_map_png_base64": overlay_b64,
            "mode": "trained_model",
        }


detector = DetectorBackend()
