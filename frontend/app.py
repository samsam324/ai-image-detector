import base64
import io

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Instrument+Serif:ital@0;1&display=swap');

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Dark gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0c0c0f 0%, #1a1a24 50%, #0f0f14 100%);
    background-attachment: fixed;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse 60% 40% at 80% 20%, rgba(168, 85, 247, 0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1400px;
    position: relative;
    z-index: 1;
}

/* Typography */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e4e4e7;
}

h1, h2, h3, .stSubheader {
    font-family: 'Instrument Serif', Georgia, serif !important;
    color: #fafafa !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 28px 32px;
    box-shadow: 
        0 4px 24px rgba(0, 0, 0, 0.4),
        0 0 0 1px rgba(255, 255, 255, 0.05) inset;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: rgba(255, 255, 255, 0.12);
    box-shadow: 
        0 8px 40px rgba(0, 0, 0, 0.5),
        0 0 0 1px rgba(255, 255, 255, 0.08) inset;
}

/* Hero Header */
.hero-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.08) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 28px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
}

.hero-kicker {
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #a78bfa;
    font-weight: 600;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.hero-kicker::before {
    content: "";
    width: 8px;
    height: 8px;
    background: #a78bfa;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.1); }
}

.hero-title {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #fafafa;
    margin: 0 0 12px 0;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.hero-subtitle {
    color: #a1a1aa;
    font-size: 1.05rem;
    line-height: 1.6;
    max-width: 700px;
}

/* Tech Pills */
.pills-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.tech-pill {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #d4d4d8;
    padding: 8px 14px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}

.tech-pill:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.2);
    color: #fafafa;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    border: 0;
    margin: 24px 0;
}

/* Section Title */
.section-title {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.5rem;
    color: #fafafa;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255, 255, 255, 0.15), transparent);
}

/* KPI Display */
.kpi-container {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
    margin-bottom: 20px;
}

.kpi-main {
    flex: 1;
}

.kpi-number {
    font-family: 'DM Sans', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
    background: linear-gradient(135deg, #fafafa 0%, #a1a1aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.kpi-label {
    color: #71717a;
    font-size: 0.9rem;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.verdict-box {
    text-align: right;
}

.verdict-text {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.4rem;
    color: #fafafa;
    margin-bottom: 6px;
}

.confidence-row {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
}

.confidence-label {
    color: #71717a;
    font-size: 0.85rem;
}

.confidence-value {
    font-weight: 700;
    font-size: 0.9rem;
    padding: 4px 12px;
    border-radius: 100px;
}

.conf-high {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.conf-medium {
    background: rgba(234, 179, 8, 0.15);
    color: #facc15;
    border: 1px solid rgba(234, 179, 8, 0.3);
}

.conf-low {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Custom Progress Bar */
.progress-track {
    height: 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 100px;
    overflow: hidden;
    position: relative;
}

.progress-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #818cf8 0%, #a78bfa 50%, #c084fc 100%);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.progress-fill::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* Hide default progress */
.stProgress { display: none; }

/* File Badges */
.badge-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

.file-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 10px 16px;
    border-radius: 12px;
    font-size: 0.85rem;
    color: #d4d4d8;
    transition: all 0.2s ease;
}

.file-badge:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.12);
}

.badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6366f1;
}

.badge-label {
    color: #71717a;
    font-weight: 500;
}

.badge-value {
    color: #fafafa;
    font-weight: 600;
}

/* Image containers */
.image-frame {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 12px;
    transition: all 0.3s ease;
}

.image-frame:hover {
    border-color: rgba(255, 255, 255, 0.15);
}

.image-label {
    color: #71717a;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.image-label::before {
    content: "";
    width: 4px;
    height: 4px;
    background: #6366f1;
    border-radius: 50%;
}

img { 
    border-radius: 10px !important;
}

/* Signals Section */
.signal-item {
    margin-bottom: 16px;
}

.signal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.signal-name {
    color: #d4d4d8;
    font-size: 0.9rem;
    font-weight: 500;
}

.signal-value {
    color: #a1a1aa;
    font-size: 0.85rem;
    font-family: 'DM Sans', monospace;
}

.signal-track {
    height: 4px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 100px;
    overflow: hidden;
}

.signal-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 100px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Upload area styling */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.02);
    border: 2px dashed rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.4);
    background: rgba(99, 102, 241, 0.05);
}

[data-testid="stFileUploader"] section {
    padding: 0;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
}

/* Info box */
.stAlert {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
    color: #a5b4fc !important;
}

/* Spinner */
.stSpinner > div {
    border-color: #6366f1 transparent transparent transparent !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}
</style>
""",
    unsafe_allow_html=True,
)

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def read_image_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def call_api_predict(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes)}
    r = requests.post(f"{API_URL}/predict", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def nice_verdict(v: str) -> str:
    if v == "likely_ai":
        return "Likely AI-Generated"
    if v == "likely_real":
        return "Likely Authentic"
    return "Uncertain"


def conf_class(conf: str) -> str:
    if conf == "High":
        return "conf-high"
    if conf == "Medium":
        return "conf-medium"
    return "conf-low"


st.markdown(
    """
<div class="hero-card">
    <div class="hero-kicker">Advanced Detection System</div>
    <h1 class="hero-title">AI Image Detector</h1>
    <p class="hero-subtitle">
        Upload an image to analyze its authenticity. This classifier uses CLIP embeddings 
        with isotonic calibration to estimate the probability of AI generation. It could
        be much better with more data and training.
    </p>
    <div class="pills-row">
        <span class="tech-pill">ViT-B-32</span>
        <span class="tech-pill">Logistic Regression</span>
        <span class="tech-pill">Isotonic Calibration</span>
        <span class="tech-pill">FastAPI Backend</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.3, 0.7], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Image Analysis</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("Upload an image to begin analysis")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    img_bytes = uploaded.getvalue()

    try:
        pil = read_image_bytes(img_bytes)
    except Exception as e:
        st.error(f"Could not read this file as an image: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            result = call_api_predict(img_bytes, uploaded.name)
        except Exception as e:
            st.error(f"Backend error: {e}\n\nMake sure FastAPI is running at {API_URL}.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

    # Decode explanation overlay
    overlay_img = None
    b64 = result.get("explanation_map_png_base64", "")
    if b64:
        try:
            overlay_png = base64.b64decode(b64)
            overlay_img = read_image_bytes(overlay_png)
        except Exception:
            overlay_img = None

    # Display images
    img_col1, img_col2 = st.columns(2, gap="medium")
    with img_col1:
        st.markdown(
            '<div class="image-frame"><div class="image-label">Original</div>',
            unsafe_allow_html=True,
        )
        st.image(pil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with img_col2:
        st.markdown(
            '<div class="image-frame"><div class="image-label">Explanation Map</div>',
            unsafe_allow_html=True,
        )
        if overlay_img is not None:
            st.image(overlay_img, use_container_width=True)
        else:
            st.caption("No explanation map available")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)

    # File info badges
    file_kb = len(img_bytes) / 1024.0
    mode_val = result.get("mode", "Standard")
    st.markdown(
        f"""
<div class="badge-row">
    <div class="file-badge">
        <span class="badge-dot"></span>
        <span class="badge-label">File</span>
        <span class="badge-value">{uploaded.name}</span>
    </div>
    <div class="file-badge">
        <span class="badge-dot"></span>
        <span class="badge-label">Size</span>
        <span class="badge-value">{file_kb:,.0f} KB</span>
    </div>
    <div class="file-badge">
        <span class="badge-dot"></span>
        <span class="badge-label">Mode</span>
        <span class="badge-value">{mode_val}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)

    if uploaded is not None:
        p = clamp01(float(result.get("ai_probability", 0.0)))
        pct = int(round(p * 100))
        verdict = nice_verdict(result.get("verdict", ""))
        conf = result.get("confidence", "Low")

        st.markdown(
            f"""
<div class="kpi-container">
    <div class="kpi-main">
        <div class="kpi-number">{pct}%</div>
        <div class="kpi-label">AI Likelihood</div>
    </div>
    <div class="verdict-box">
        <div class="verdict-text">{verdict}</div>
        <div class="confidence-row">
            <span class="confidence-label">Confidence</span>
            <span class="confidence-value {conf_class(conf)}">{conf}</span>
        </div>
    </div>
</div>

<div class="progress-track">
    <div class="progress-fill" style="width: {pct}%;"></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider" />', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title" style="font-size: 1.1rem;">Analysis Signals</div>',
            unsafe_allow_html=True,
        )

        signals = result.get("signals") or {}
        if not signals:
            st.caption("No additional signals returned.")
        else:
            for k, v in signals.items():
                val = clamp01(float(v))
                val_pct = int(round(val * 100))
                st.markdown(
                    f"""
<div class="signal-item">
    <div class="signal-header">
        <span class="signal-name">{k}</span>
        <span class="signal-value">{val_pct}%</span>
    </div>
    <div class="signal-track">
        <div class="signal-fill" style="width: {val_pct}%;"></div>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)