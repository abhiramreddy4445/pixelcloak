<![CDATA[<div align="center">

# PixelCloak

### Adversarial Privacy Filter for Facial Recognition Defense

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00e5ff?style=for-the-badge)

*Imperceptible noise. Invisible to humans. Devastating to models.*

[Features](#features) · [How It Works](#how-it-works) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Tech Stack](#tech-stack)

---

</div>

## The Problem

Facial recognition systems are deployed at airports, offices, and across social media. They raise serious privacy concerns — people should have the ability to opt out of being identified. Simple approaches like blurring destroy the photo and are easily defeated by modern models trained on noisy data.

## The Solution

PixelCloak uses **adversarial perturbation** — a mathematically precise attack on the neural network itself. Instead of destroying image quality, it exploits the model's own gradient structure to add noise that is:

- **Imperceptible** — bounded by an L∞ norm (ε), no pixel changes more than ~7/255
- **Effective** — causes the model's face embedding to collapse (cosine similarity → 0)
- **Provable** — verification engine compares embeddings before and after

## Features

| Feature | Description |
|---|---|
| **FGSM Attack** | Fast Gradient Sign Method — single-step perturbation using the sign of the input gradient |
| **PGD Attack** | Projected Gradient Descent — iterative multi-step attack, strictly stronger than FGSM |
| **L∞ Bounded** | Perturbation mathematically constrained by epsilon — imperceptible to humans |
| **Face Detection** | MTCNN-based automatic face detection and cropping to 160×160 |
| **Embedding Verification** | Cosine similarity between original and adversarial embeddings proves the attack works |
| **Real-time UI** | Streamlit interface with attack controls, side-by-side comparison, and download |

## How It Works

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

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/yourusername/PixelCloak.git
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

## Why Not Just Blur?

| | Blurring | Adversarial Perturbation |
|---|---|---|
| **Visible?** | Yes, obviously | No, imperceptible |
| **Model still works?** | Partially, yes | No, embedding collapses |
| **Mathematical guarantee?** | None | L∞ bounded by ε |
| **Photo still usable?** | Degraded | Looks identical |

Blurring is a privacy illusion. Modern face recognition models are trained on noisy, low-res, blurry images — they're robust to blur by design. Adversarial perturbation exploits the model's own gradient structure to push the embedding across a decision boundary while changing almost nothing visually.

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with PyTorch · FaceNet (InceptionResnetV1) · Streamlit · MTCNN**

</div>
]]>