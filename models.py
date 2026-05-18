"""
models.py — Victim Model Loader for PixelCloak.

Loads FaceNet (InceptionResnetV1) pretrained on VGGFace2 as the "victim"
facial recognition model, plus MTCNN for face detection.

Key design: the model is ALWAYS in eval mode and parameters are FROZEN.
We never update model weights — we only use it to compute gradients
w.r.t. the input image.
"""

import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image


def get_device() -> torch.device:
    """Return CPU device by default (safe for any hardware)."""
    return torch.device("cpu")


def load_facenet(device: torch.device | None = None) -> nn.Module:
    """
    Load InceptionResnetV1 pretrained on VGGFace2.

    Architecture:
    - Input:  (batch, 3, 160, 160) — RGB face crop, normalized to [-1, 1]
    - Output: (batch, 512) — L2-normalized embedding vector

    The embedding lives on a hypersphere: ||e||₂ = 1.
    Two embeddings of the same person → cosine similarity ≈ 1.0
    Different people → cosine similarity ≈ 0.0–0.5
    """
    if device is None:
        device = get_device()

    try:
        model = InceptionResnetV1(pretrained="vggface2", classify=False)
    except Exception as e:
        raise RuntimeError(
            "Failed to download FaceNet pretrained weights. "
            "Check your internet connection or manually download VGGFace2 weights. "
            f"Original error: {e}"
        ) from e

    model.eval()  # Disable dropout/batchnorm training behavior
    model.to(device)

    # Freeze all parameters — we will NOT update model weights
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_mtcnn(device: torch.device | None = None) -> MTCNN:
    """
    Load MTCNN for face detection and cropping.

    MTCNN (Multi-task Cascaded Convolutional Networks) detects faces and
    returns cropped + aligned 160×160 face regions ready for FaceNet.
    """
    if device is None:
        device = get_device()

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,  # Return only the most prominent face
        device=device,
    )
    return mtcnn


def detect_and_crop_face(
    image: Image.Image, mtcnn: MTCNN
) -> tuple[Image.Image | None, list | None]:
    """
    Detect the primary face and return the cropped 160×160 face.

    Returns:
        (cropped_face_pil, bounding_box) or (None, None) if no face found.
    """
    boxes, probs = mtcnn.detect(image)

    if boxes is None or len(boxes) == 0:
        return None, None

    # MTCNN returns a (3, 160, 160) tensor in [-1, 1]
    face_tensor = mtcnn(image)

    if face_tensor is None:
        return None, None

    # Convert tensor back to PIL for display
    face_np = face_tensor.detach().cpu().numpy()
    face_np = np.transpose(face_np, (1, 2, 0))  # (H, W, C)
    face_np = ((face_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    face_pil = Image.fromarray(face_np)

    return face_pil, boxes[0].tolist()


def get_embedding(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Get 512-dim L2-normalized face embedding."""
    with torch.no_grad():
        return model(tensor)
