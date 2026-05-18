"""
utils.py — Image ↔ Tensor Conversion Utilities for PixelCloak.

Handles the bridge between PIL images and PyTorch tensors,
including the specific preprocessing that FaceNet (InceptionResnetV1)
expects: pixel values normalized to [-1, 1].
"""

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor suitable for FaceNet.

    FaceNet expects:
    - Shape: (1, 3, H, W) — batch dim + channels-first
    - Range:  [-1, 1] — pixel values normalized from [0, 255]

    The formula: tensor = (pixels / 127.5) - 1.0
        - 0     → -1.0
        - 127.5 →  0.0
        - 255   → +1.0
    """
    # PIL → numpy float32 in [0, 255]
    arr = np.array(image, dtype=np.float32)
    # [0, 255] → [0, 1] → [-1, 1]
    arr = arr / 127.5 - 1.0
    # (H, W, C) → (C, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    # Add batch dim: (1, C, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a FaceNet-format tensor back to a PIL Image.

    Reverses pil_to_tensor:
    - Removes batch dim
    - (C, H, W) → (H, W, C)
    - [-1, 1] → [0, 255] uint8
    """
    # Remove batch dim → (C, H, W)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # [-1, 1] → [0, 1]
    tensor = (tensor + 1.0) / 2.0
    # Clamp to valid range
    tensor = tensor.clamp(0, 1)
    # (C, H, W) → (H, W, C) → numpy
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor_to_noise_map(tensor: torch.Tensor) -> Image.Image:
    """
    Visualize the perturbation noise as an amplified image.

    The raw noise is tiny (e.g., ε=0.03 means max ±7.6/255 per pixel).
    We amplify it by 10× and center at gray so the user can see the pattern.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # Amplify for visibility, center at gray (0.5)
    tensor = tensor * 10.0 + 0.5
    tensor = tensor.clamp(0, 1)
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_noise_magnitude(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Return the L∞ norm: max absolute pixel change."""
    return torch.max(torch.abs(adversarial - original)).item()
