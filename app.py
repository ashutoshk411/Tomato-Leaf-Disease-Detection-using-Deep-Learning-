import streamlit as st
import numpy as np
import cv2
from PIL import Image

from model import predict, CLASS_NAMES
from utils import (
    estimate_severity,
    check_image_quality,
    consensus_predict,
    get_recommendation,
    get_disease_info,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="centered",
)

st.markdown("""
<style>
    .main { max-width: 780px; }
    .disease-card {
        background: #f9fdf9;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 5px solid #f9a825;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .error-box {
        background: #ffebee;
        border-left: 5px solid #c62828;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .severity-mild     { color: #2e7d32; font-weight: bold; }
    .severity-moderate { color: #e65100; font-weight: bold; }
    .severity-severe   { color: #b71c1c; font-weight: bold; }
    .severity-none     { color: #1565c0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🍅 Tomato Leaf Disease Detector")
st.caption("Upload 1–3 photos of a tomato leaf to detect disease, estimate severity, and get treatment advice.")

# ── Model status ──────────────────────────────────────────────────────────────
import os
if not os.path.exists("tomato_model.h5"):
    st.markdown("""
<div class="warning-box">
⚠️ <b>No trained model found (tomato_model.h5).</b><br>
The app will build the model architecture but predictions will be random until you train it.<br>
Run <code>python model.py</code> or call <code>train_model(data_dir)</code> from <code>model.py</code> to train.
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload leaf image(s)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Upload 1 to 3 images. If multiple images are uploaded, majority voting is used.",
)

if not uploaded_files:
    st.info("👆 Upload a tomato leaf image to get started.")
    st.stop()

if len(uploaded_files) > 3:
    st.warning("Please upload a maximum of 3 images. Only the first 3 will be used.")
    uploaded_files = uploaded_files[:3]

# ── Process each image ────────────────────────────────────────────────────────
def load_rgb(file):
    img = Image.open(file).convert("RGB")
    return np.array(img)

all_predictions = []
results_per_image = []

cols = st.columns(len(uploaded_files))
for i, f in enumerate(uploaded_files):
    img_rgb = load_rgb(f)

    with cols[i]:
        st.image(img_rgb, caption=f.name, use_container_width=True)

    # Quality check
    quality_ok, quality_warnings = check_image_quality(img_rgb)

    # Predict
    disease, confidence, probs = predict(img_rgb)

    # Severity (skip for healthy)
    if disease == "Healthy":
        severity, infected_pct = "None", 0.0
    else:
        severity, infected_pct = estimate_severity(img_rgb)

    all_predictions.append((disease, confidence, probs))
    results_per_image.append({
        "name": f.name,
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "infected_pct": infected_pct,
        "quality_ok": quality_ok,
        "quality_warnings": quality_warnings,
    })

# ── Consensus (multi-image) ───────────────────────────────────────────────────
if len(all_predictions) > 1:
    final_disease, vote_note = consensus_predict(all_predictions)
    # Use the result entry whose disease matches the consensus for severity
    matching = [r for r in results_per_image if r["disease"] == final_disease]
    best_result = max(matching, key=lambda x: x["confidence"])
    st.markdown(f"### 🗳️ Consensus Result: **{final_disease}**")
    st.caption(vote_note)
else:
    best_result = results_per_image[0]
    final_disease = best_result["disease"]

st.divider()

# ── Main result card ──────────────────────────────────────────────────────────
severity = best_result["severity"]
confidence = best_result["confidence"]
infected_pct = best_result["infected_pct"]

# Quality warnings
for r in results_per_image:
    for w in r["quality_warnings"]:
        st.markdown(f'<div class="warning-box">⚠️ <b>{r["name"]}:</b> {w}</div>', unsafe_allow_html=True)

# Low confidence warning
if confidence < 0.60:
    st.markdown(
        '<div class="warning-box">⚠️ <b>Low confidence prediction.</b> '
        'Please upload a clearer, well-lit, close-up image of the leaf.</div>',
        unsafe_allow_html=True,
    )

severity_class = {
    "Mild": "severity-mild",
    "Moderate": "severity-moderate",
    "Severe": "severity-severe",
    "None": "severity-none",
}.get(severity, "severity-none")

col1, col2, col3 = st.columns(3)
col1.metric("🦠 Disease", final_disease)
col2.metric("📊 Confidence", f"{confidence * 100:.1f}%")
if severity != "None":
    col3.metric("🌿 Severity", f"{severity} ({infected_pct:.1f}%)")
else:
    col3.metric("🌿 Severity", "None")

# Confidence bar
st.progress(min(confidence, 1.0), text=f"Prediction confidence: {confidence * 100:.1f}%")

# ── Per-image breakdown (multi-image only) ────────────────────────────────────
if len(results_per_image) > 1:
    with st.expander("📋 Per-image predictions"):
        for r in results_per_image:
            sev_str = f"{r['severity']} ({r['infected_pct']:.1f}%)" if r["severity"] != "None" else "None"
            st.write(
                f"**{r['name']}** → {r['disease']} | "
                f"Confidence: {r['confidence']*100:.1f}% | Severity: {sev_str}"
            )

st.divider()

# ── Recommendation ────────────────────────────────────────────────────────────
rec = get_recommendation(final_disease, severity)
st.markdown("### 💊 Treatment Recommendation")
st.markdown(f'<div class="disease-card">{rec}</div>', unsafe_allow_html=True)

# ── Disease info panel ────────────────────────────────────────────────────────
info = get_disease_info(final_disease)
with st.expander(f"ℹ️ About {final_disease}", expanded=False):
    st.markdown(f"**Description**\n\n{info['description']}")
    st.markdown(f"**Causes**\n\n{info['causes']}")
    st.markdown("**Prevention Tips**")
    for tip in info["prevention"]:
        st.markdown(f"- {tip}")

# ── Probability breakdown ─────────────────────────────────────────────────────
with st.expander("📊 Full prediction probabilities", expanded=False):
    # Use the result with best confidence for display
    _, _, probs = all_predictions[0] if len(all_predictions) == 1 else max(all_predictions, key=lambda x: x[1])
    for cls, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        st.write(f"{cls}")
        st.progress(float(prob), text=f"{prob * 100:.1f}%")
