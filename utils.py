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


def apply_noise_to_original(
    original: Image.Image,
    noise_tensor: torch.Tensor,
    bbox: list,
) -> Image.Image:
    """
    Apply the adversarial noise pattern directly to the face region of the
    original full-size image. Uses the tight bounding box (no margin)
    so only the face area gets perturbed.

    Args:
        original: The full original image (any size).
        noise_tensor: (1, 3, 160, 160) perturbation from the attack.
        bbox: [x1, y1, x2, y2] bounding box from MTCNN detection.

    Returns:
        Full-size PIL image with noise applied to the face region only.
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    w, h = original.size

    # Clamp to image bounds (no margin expansion)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    box_w, box_h = x2 - x1, y2 - y1

    if box_w <= 0 or box_h <= 0:
        return original.copy()

    # Get noise as numpy array in [-1, 1] range, shape (H, W, C)
    noise = noise_tensor.squeeze(0).detach().cpu().numpy()
    noise = np.transpose(noise, (1, 2, 0))  # (C,H,W) -> (H,W,C)

    # Convert noise from [-1,1] tensor space to pixel space [-127.5, 127.5]
    noise_pixels = noise * 127.5

    # Resize noise to match the bounding box dimensions
    noise_pil = Image.fromarray(((noise_pixels + 127.5)).clip(0, 255).astype(np.uint8))
    noise_resized = noise_pil.resize((box_w, box_h), Image.BILINEAR)
    noise_arr = np.array(noise_resized, dtype=np.float32) - 127.5

    # Apply noise to the face region of the original image
    result = original.copy()
    orig_arr = np.array(result, dtype=np.float32)
    face_region = orig_arr[y1:y2, x1:x2, :]

    # Add noise and clip to valid range
    face_region = (face_region + noise_arr).clip(0, 255)
    orig_arr[y1:y2, x1:x2, :] = face_region

    return Image.fromarray(orig_arr.astype(np.uint8))


def compute_noise_magnitude(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Return the L∞ norm: max absolute pixel change."""
    return torch.max(torch.abs(adversarial - original)).item()
