# PixelCloak

### Adversarial Privacy Filter for Facial Recognition Defense

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00e5ff?style=for-the-badge)

*Imperceptible noise. Invisible to humans. Devastating to models.*

---

## The Problem

Facial recognition systems are deployed at airports, offices, and across social media. They raise serious privacy concerns — people should have the ability to opt out of being identified.

The obvious answer? Blur the photo. But that doesn't actually work.

## Why Blurring Doesn't Work

Modern face recognition models are **trained on noisy, low-res, blurry images**. They're robust to blur by design. You'd have to blur so much that the photo becomes useless for humans too — you've destroyed the image to protect it.

```
Blurred image:
  Human sees:  "This is blurry, clearly tampered"
  Model sees:  "Still probably John, 70% confidence"

Adversarial image (PixelCloak):
  Human sees:  "This is John"  (looks completely normal)
  Model sees:  "This is absolutely not John, 0.02% confidence"
```

| | Blurring | Adversarial Perturbation |
|---|---|---|
| **Visible to humans?** | Yes, obviously | No, imperceptible |
| **Model still works?** | Partially, yes | No — embedding collapses |
| **Mathematical guarantee?** | None | L∞ bounded by ε |
| **Photo still usable?** | Degraded | Looks identical |

Think of it this way: **blurring is like putting a mask on someone's face — obvious to everyone. Adversarial perturbation is like changing their fingerprint — invisible to the naked eye, but the scanner reads a completely different identity.**

## How PixelCloak Works

PixelCloak doesn't destroy image quality. It exploits the **mathematical structure** of the neural network. Because face recognition models compute gradients, we can compute exactly how to minimally perturb the input to maximally change the output.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL ATTACK PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Normal ML Training:        Adversarial Attack:                 │
│  ∂L/∂W → update weights     ∂L/∂x → update pixels              │
│                                                                 │
│  1. Feed image x through frozen FaceNet → embedding e_orig      │
│  2. Compute loss = MSE(e_adv, e_orig) — maximize distance       │
│  3. Backprop to get ∂L/∂x — gradient w.r.t. PIXELS              │
│  4. Perturb: x_adv = x + ε · sign(∂L/∂x)                       │
│  5. Clip: |x_adv - x| ≤ ε  (L∞ constraint)                     │
│  6. Verify: cosine_similarity(e_orig, e_adv) ≈ 0                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**FGSM** does this in one step. **PGD** iterates it with smaller steps, projecting back into the ε-ball after each iteration — a constrained optimization that finds stronger perturbations.

## Features

| Feature | Description |
|---|---|
| **FGSM Attack** | Fast Gradient Sign Method — single-step perturbation using the sign of the input gradient |
| **PGD Attack** | Projected Gradient Descent — iterative multi-step attack, strictly stronger than FGSM |
| **L∞ Bounded** | Perturbation mathematically constrained by epsilon — imperceptible to humans |
| **Face Detection** | MTCNN-based automatic face detection and cropping to 160×160 |
| **Embedding Verification** | Cosine similarity between original and adversarial embeddings proves the attack works |
| **Real-time UI** | Streamlit interface with attack controls, side-by-side comparison, and download |

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/abhiramreddy4445/PixelCloak.git
cd PixelCloak
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload a photo, adjust epsilon, and click **Run Adversarial Attack**.

### Docker

```bash
docker build -t pixelcloak .
docker run -p 8501:8501 pixelcloak
```

## Architecture

```
PixelCloak/
├── app.py              # Streamlit UI — upload, adjust ε, side-by-side comparison
├── attacks.py          # FGSM and PGD implementations with gradient math
├── models.py           # FaceNet + MTCNN loader (frozen victim model)
├── utils.py            # PIL ↔ tensor conversions, noise visualization
├── requirements.txt    # Python dependencies
├── Dockerfile          # Containerized deployment
├── .streamlit/
│   └── config.toml     # Custom dark theme
├── LICENSE             # MIT
└── README.md
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `models.py` | Loads InceptionResnetV1 (VGGFace2) and MTCNN. Model is frozen — parameters never update. |
| `attacks.py` | Pure math. FGSM and PGD compute gradients w.r.t. input pixels, not model weights. |
| `utils.py` | Image ↔ tensor conversion with [-1, 1] normalization. Noise visualization. |
| `app.py` | Streamlit frontend. Orchestrates detection, attack, verification, and display. |

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **ML Framework** | PyTorch | Autograd for input gradient computation |
| **Victim Model** | InceptionResnetV1 (VGGFace2) | Production-grade face recognition, 512-dim embeddings |
| **Face Detection** | MTCNN | Automatic face cropping and alignment |
| **Frontend** | Streamlit | Rapid prototyping with interactive widgets |
| **Attack Math** | Custom FGSM/PGD | From-scratch implementation with educational comments |

## License

MIT — see [LICENSE](LICENSE) for details.

---

**Built with PyTorch · FaceNet (InceptionResnetV1) · Streamlit · MTCNN**
