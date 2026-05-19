"""
app.py — PixelCloak: Adversarial Privacy Filter

A professional Streamlit interface that applies imperceptible adversarial
perturbations to face images, causing facial recognition models to misidentify
the subject while leaving the image visually unchanged.

Usage:
    streamlit run app.py
"""

import io
import streamlit as st
import torch
import numpy as np
from PIL import Image

from models import load_facenet, load_mtcnn, detect_and_crop_face, get_device
from attacks import fgsm_attack, pgd_attack
from utils import pil_to_tensor, tensor_to_pil, tensor_to_noise_map, compute_noise_magnitude, apply_noise_to_original

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PixelCloak — Adversarial Privacy Filter",
    page_icon="assets/favicon.png" if __import__("os").path.exists("assets/favicon.png") else " ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Hero header ── */
    .hero {
        background: linear-gradient(135deg, #0a0e17 0%, #1a1a2e 50%, #16213e 100%);
        border: 1px solid #00e5ff22;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00e5ff, #7c3aed, #00e5ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        margin-bottom: 0.3rem;
    }
    @keyframes shimmer {
        to { background-position: 200% center; }
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin: 0;
    }
    .hero .tagline {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: 0.8rem;
        font-style: italic;
    }

    /* ── Feature cards (empty state) ── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .feature-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        transition: border-color 0.2s;
    }
    .feature-card:hover {
        border-color: #00e5ff44;
    }
    .feature-card h3 {
        font-size: 1rem;
        color: #00e5ff;
        margin: 0 0 0.5rem 0;
    }
    .feature-card p {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #111827;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-effective { border: 2px solid #10b981; }
    .metric-effective .value { color: #10b981; }
    .metric-partial  { border: 2px solid #f59e0b; }
    .metric-partial .value  { color: #f59e0b; }
    .metric-weak     { border: 2px solid #ef4444; }
    .metric-weak .value     { color: #ef4444; }
    .metric-neutral  { border: 2px solid #334155; }
    .metric-neutral .value  { color: #e2e8f0; }

    /* ── Image containers ── */
    .img-container {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .img-container .caption {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.5rem;
    }

    /* ── Verdict banner ── */
    .verdict-banner {
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .verdict-effective {
        background: #10b98115;
        border: 1px solid #10b98144;
        color: #6ee7b7;
    }
    .verdict-partial {
        background: #f59e0b15;
        border: 1px solid #f59e0b44;
        color: #fcd34d;
    }
    .verdict-weak {
        background: #ef444415;
        border: 1px solid #ef444444;
        color: #fca5a5;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        border-top: 1px solid #1e293b;
        margin-top: 3rem;
    }
    .footer a {
        color: #00e5ff;
        text-decoration: none;
        margin: 0 0.8rem;
    }
    .footer a:hover { text-decoration: underline; }
    .footer .tech {
        color: #475569;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }

    /* ── Sidebar polish ── */
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #cbd5e1 !important;
    }

    /* ── Remove default Streamlit padding ── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* ── How it works section ── */
    .how-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin: 0.8rem 0;
    }
    .how-step .num {
        background: #00e5ff22;
        color: #00e5ff;
        border-radius: 50%;
        min-width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .how-step .text {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load models (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = get_device()
    model = load_facenet(device)
    mtcnn = load_mtcnn(device)
    return model, mtcnn, device


model, mtcnn, device = load_models()


# ──────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>PixelCloak</h1>
    <p>Adversarial Privacy Filter for Facial Recognition Defense</p>
    <div class="tagline">Imperceptible noise. Invisible to humans. Devastating to models.</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Attack Parameters")

    attack_type = st.selectbox(
        "Method",
        ["FGSM", "PGD"],
        help="FGSM = single-step (fast). PGD = iterative (stronger).",
    )

    epsilon = st.slider(
        "Epsilon ( \u03b5 )",
        min_value=0.005,
        max_value=0.15,
        value=0.03,
        step=0.005,
        help="L\u221e bound on per-pixel perturbation. 0.01 = subtle, 0.03 = sweet spot, 0.10+ = visible.",
    )

    if attack_type == "PGD":
        pgd_steps = st.slider(
            "PGD Iterations",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
        )
        pgd_alpha = st.slider(
            "Step Size ( \u03b1 )",
            min_value=0.001,
            max_value=round(epsilon / 2, 4),
            value=min(0.005, round(epsilon / 4, 4)),
            step=0.001,
        )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.82rem; color:#64748b; line-height:1.6;">
        <strong style="color:#94a3b8;">How it works</strong><br>
        The attack computes gradients of the face embedding
        w.r.t. input pixels, then perturbs each pixel in the
        direction that maximally shifts the embedding
        &mdash; causing the model to misidentify the face.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image containing a face",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="The system will detect the primary face and apply adversarial perturbation.",
)

if uploaded_file is None:
    # ── Empty state: feature cards ──
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <h3>FGSM Attack</h3>
            <p>Fast Gradient Sign Method &mdash; single-step perturbation using the sign of the input gradient. Fast and effective.</p>
        </div>
        <div class="feature-card">
            <h3>PGD Attack</h3>
            <p>Projected Gradient Descent &mdash; iterative multi-step attack with projection back into the epsilon ball. Stronger than FGSM.</p>
        </div>
        <div class="feature-card">
            <h3>L\u221e Bounded</h3>
            <p>Perturbation is mathematically bounded by epsilon. No pixel changes more than the threshold &mdash; imperceptible to humans.</p>
        </div>
        <div class="feature-card">
            <h3>Embedding Verification</h3>
            <p>Cosine similarity between original and adversarial embeddings proves the model sees a different person.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How PixelCloak Works", expanded=False):
        st.markdown("""
        <div class="how-step"><div class="num">1</div><div class="text"><strong>Upload</strong> a photo containing a face.</div></div>
        <div class="how-step"><div class="num">2</div><div class="text"><strong>Detect</strong> the face using MTCNN and crop to 160&times;160.</div></div>
        <div class="how-step"><div class="num">3</div><div class="text"><strong>Compute gradients</strong> of the face embedding w.r.t. each input pixel.</div></div>
        <div class="how-step"><div class="num">4</div><div class="text"><strong>Perturb</strong> pixels in the direction that maximally changes the embedding, bounded by &epsilon;.</div></div>
        <div class="how-step"><div class="num">5</div><div class="text"><strong>Verify</strong> by comparing cosine similarity of original vs adversarial embedding.</div></div>
        """, unsafe_allow_html=True)

        st.code("""
# The core math (simplified):
e_orig = model(x)                        # Original embedding
loss   = MSE(model(x_adv), e_orig)       # Maximize embedding distance
grad   = \u2202loss/\u2202x_adv                    # Gradient w.r.t. PIXELS (not weights)
x_adv  = x + \u03b5 \u00b7 sign(grad)                # Perturb each pixel
x_adv  = clamp(x_adv, x-\u03b5, x+\u03b5)            # L\u221e constraint
        """, language="python")

else:
    # ── Image uploaded: process it ──
    original_image = Image.open(uploaded_file).convert("RGB")

    col_img, col_crop = st.columns([3, 1])
    with col_img:
        st.markdown("**Original Image**")
        st.image(original_image, use_container_width=True)

    # Detect face
    with st.spinner("Detecting face..."):
        cropped_face, bbox = detect_and_crop_face(original_image, mtcnn)

    if cropped_face is None:
        st.error("No face detected. Please upload a clearer photo with a visible face.")
        st.stop()

    with col_crop:
        st.markdown("**Detected Face**")
        st.image(cropped_face, use_container_width=True)
        st.caption("160 x 160 crop")

    # ── Run attack ──
    if st.button("  Run Adversarial Attack", type="primary", use_container_width=True):
        face_tensor = pil_to_tensor(cropped_face).to(device)

        if attack_type == "FGSM":
            with st.spinner("Running FGSM attack..."):
                adv_tensor, noise_tensor, cosine_sim = fgsm_attack(
                    model, face_tensor, epsilon=epsilon
                )
        else:
            # PGD with progress bar
            progress_bar = st.progress(0, text="Running PGD attack...")
            step_count = [0]

            def on_step(current, total):
                step_count[0] = current
                progress_bar.progress(
                    current / total,
                    text=f"PGD iteration {current}/{total}",
                )

            adv_tensor, noise_tensor, cosine_sim = pgd_attack(
                model, face_tensor,
                epsilon=epsilon,
                alpha=pgd_alpha,
                num_steps=pgd_steps,
                progress_callback=on_step,
            )
            progress_bar.progress(1.0, text="PGD complete!")

        adv_image = tensor_to_pil(adv_tensor)
        noise_image = tensor_to_noise_map(noise_tensor)
        l_inf = compute_noise_magnitude(face_tensor, adv_tensor)

        # Apply adversarial noise to the face region of the original image
        full_adv_image = apply_noise_to_original(original_image, noise_tensor, bbox)


        # ── Results section ──
        st.markdown("---")
        st.markdown("### Results")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown('<div class="img-container"><div class="caption">Original Image</div></div>', unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)
        with r2:
            st.markdown('<div class="img-container"><div class="caption">Adversarial Image (face perturbed)</div></div>', unsafe_allow_html=True)
            st.image(full_adv_image, use_container_width=True)

        r3, r4, r5 = st.columns(3)
        with r3:
            st.markdown('<div class="img-container"><div class="caption">Original Face (160x160)</div></div>', unsafe_allow_html=True)
            st.image(cropped_face, use_container_width=True)
        with r4:
            st.markdown('<div class="img-container"><div class="caption">Adversarial Face (160x160)</div></div>', unsafe_allow_html=True)
            st.image(adv_image, use_container_width=True)
        with r5:
            st.markdown('<div class="img-container"><div class="caption">Noise Pattern (10x amplified)</div></div>', unsafe_allow_html=True)
            st.image(noise_image, use_container_width=True)

        # ── Metrics ──
        st.markdown("")
        m1, m2, m3 = st.columns(3)

        if cosine_sim < 0.5:
            verdict_class = "metric-effective"
            verdict_label = "EFFECTIVE"
        elif cosine_sim < 0.7:
            verdict_class = "metric-partial"
            verdict_label = "PARTIAL"
        else:
            verdict_class = "metric-weak"
            verdict_label = "WEAK"

        with m1:
            st.markdown(f"""
            <div class="metric-card metric-neutral">
                <div class="label">Cosine Similarity</div>
                <div class="value">{cosine_sim:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card metric-neutral">
                <div class="label">L\u221e Norm</div>
                <div class="value">{l_inf:.6f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card {verdict_class}">
                <div class="label">Attack Verdict</div>
                <div class="value">{verdict_label}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Verdict banner ──
        if cosine_sim < 0.5:
            st.markdown(f"""
            <div class="verdict-banner verdict-effective">
                <strong>Attack is effective.</strong> Cosine similarity dropped to {cosine_sim:.4f} &mdash;
                the facial recognition model would identify this as a <strong>different person</strong>.
                The perturbation is imperceptible at &epsilon;={epsilon}.
            </div>
            """, unsafe_allow_html=True)
        elif cosine_sim < 0.7:
            st.markdown(f"""
            <div class="verdict-banner verdict-partial">
                <strong>Partial effect.</strong> Similarity is {cosine_sim:.4f} &mdash;
                model confidence is reduced but it may still recognize the face.
                Try increasing &epsilon; or switching to PGD with more steps.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-banner verdict-weak">
                <strong>Weak attack.</strong> Similarity is {cosine_sim:.4f} &mdash;
                the model still recognizes the face. Increase &epsilon; or use PGD.
            </div>
            """, unsafe_allow_html=True)

        # ── Download button ──
        st.markdown("")
        buf = io.BytesIO()
        full_adv_image.save(buf, format="PNG")
        st.download_button(
            label="  Download Adversarial Image (Full)",
            data=buf.getvalue(),
            file_name="pixelcloak_adversarial.png",
            mime="image/png",
            use_container_width=True,
        )

        # ── Technical details ──
        with st.expander("Technical Details"):
            st.markdown(f"""
            | Parameter | Value |
            |---|---|
            | Attack method | {attack_type} |
            | Epsilon (L\u221e bound) | {epsilon} |
            | {"PGD steps" if attack_type == "PGD" else "Steps"} | {pgd_steps if attack_type == "PGD" else 1} |
            | {"PGD alpha" if attack_type == "PGD" else "Alpha"} | {pgd_alpha if attack_type == "PGD" else "N/A"} |
            | Cosine similarity | {cosine_sim:.6f} |
            | L\u221e norm | {l_inf:.6f} |
            | Embedding dimension | 512 |
            | Victim model | InceptionResnetV1 (VGGFace2) |
            | Face detector | MTCNN |
            """)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div>
        <a href="https://github.com" target="_blank">GitHub</a>
        <a href="https://linkedin.com" target="_blank">LinkedIn</a>
    </div>
    <div class="tech">
        Built with PyTorch &middot; FaceNet (InceptionResnetV1) &middot; Streamlit &middot; MTCNN
    </div>
</div>
""", unsafe_allow_html=True)
