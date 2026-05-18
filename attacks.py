"""
attacks.py — Adversarial Attack Implementations for PixelCloak.

This is the mathematical core of the system. We implement two attacks:
1. FGSM (Fast Gradient Sign Method) — single-step attack
2. PGD  (Projected Gradient Descent)  — iterative, stronger attack

╔══════════════════════════════════════════════════════════════════════════╗
║  KEY CONCEPT: GRADIENTS W.R.T. THE IMAGE, NOT THE MODEL WEIGHTS        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  In normal training, we compute:                                         ║
║      ∂L/∂W  (gradient of loss w.r.t. model weights W)                   ║
║      W_new = W - α · ∂L/∂W  (update weights to minimize loss)           ║
║                                                                          ║
║  In adversarial attacks, we compute:                                     ║
║      ∂L/∂x  (gradient of loss w.r.t. INPUT IMAGE PIXELS x)              ║
║      x_adv = x + ε · sign(∂L/∂x)  (update pixels to MAXIMIZE loss)      ║
║                                                                          ║
║  The model weights are FROZEN. We are asking:                            ║
║  "Which direction should I nudge each pixel to make the model's          ║
║   output change the most?"                                               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    original_tensor: torch.Tensor,
    epsilon: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Fast Gradient Sign Method (FGSM) — Goodfellow et al., 2014.

    FGSM is a SINGLE-STEP attack. It computes the gradient once and applies
    the perturbation in one shot.

    Mathematical formulation
    ────────────────────────
    Given:
        x      = original image (tensor, requires_grad=True)
        e_orig = embedding of x from frozen model
        L      = MSE(e_adv, e_orig)  — loss we want to MAXIMIZE

    Step 1: Forward pass
        e_adv = model(x)           — compute embedding of current image

    Step 2: Compute loss
        L = ||e_adv - e_orig||²₂   — MSE between embeddings

    Step 3: Backward pass (THE KEY STEP)
        ∂L/∂x = torch.autograd.grad(L, x)
        This gives us a gradient the SAME SHAPE as the image (e.g., 3×160×160).
        Each element tells us: "if I increase this pixel by a tiny amount,
        how much does the loss change?"

    Step 4: Sign of gradient
        sign(∂L/∂x) ∈ {-1, 0, +1} for each pixel
        The sign tells us the DIRECTION to push each pixel to increase loss.

    Step 5: Perturbation
        δ = ε · sign(∂L/∂x)
        x_adv = x + δ

    Step 6: Clip to valid range
        x_adv = clamp(x_adv, -1, 1)       — valid pixel range for FaceNet
        x_adv = clamp(x_adv, x-ε, x+ε)    — L∞ constraint (imperceptibility)

    Args:
        model:           Frozen FaceNet model
        original_tensor: Input image tensor, shape (1, 3, 160, 160), range [-1, 1]
        epsilon:         Maximum perturbation per pixel (L∞ bound)
                         ε=0.01 → very subtle, may not fool model
                         ε=0.03 → sweet spot, imperceptible but effective
                         ε=0.10 → visible noise, very effective

    Returns:
        adversarial_tensor: Perturbed image
        noise:              The added perturbation (for visualization)
        cosine_sim:         Cosine similarity between original and adversarial embeddings
    """
    model.eval()

    # ── Step 0: Get the ORIGINAL embedding (fixed reference point) ────────
    with torch.no_grad():
        e_orig = model(original_tensor).detach()

    # ── Step 1: Random initialization within the ε-ball ──────────────────
    # The gradient at the clean image is near-zero (flat loss surface).
    # Starting from a random point gives us a meaningful gradient direction.
    x_adv = original_tensor.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, -1.0, 1.0)
    x_adv = torch.max(torch.min(x_adv, original_tensor + epsilon), original_tensor - epsilon)

    # ── Step 2: Forward pass with gradient tracking ──────────────────────
    x_adv = x_adv.detach().requires_grad_(True)
    e_adv = model(x_adv)

    # ── Step 3: Compute loss and gradient ∂L/∂x ─────────────────────────
    loss = nn.functional.mse_loss(e_adv, e_orig)
    grad = torch.autograd.grad(
        outputs=loss,
        inputs=x_adv,
        create_graph=False,
    )[0]

    # ── Step 4: FGSM perturbation ────────────────────────────────────────
    perturbation = epsilon * grad.sign()

    # ── Step 5: Apply perturbation and clip ──────────────────────────────
    x_adv = original_tensor.detach() + perturbation
    x_adv = torch.max(torch.min(x_adv, original_tensor + epsilon), original_tensor - epsilon)
    x_adv = torch.clamp(x_adv, -1.0, 1.0)

    # ── Step 6: Verify the attack worked ─────────────────────────────────
    noise = x_adv - original_tensor.detach()
    with torch.no_grad():
        e_adv_final = model(x_adv)
        cosine_sim = nn.functional.cosine_similarity(e_orig, e_adv_final).item()

    return x_adv, noise, cosine_sim


def pgd_attack(
    model: nn.Module,
    original_tensor: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.005,
    num_steps: int = 20,
    progress_callback=None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Projected Gradient Descent (PGD) — Madry et al., 2017.

    PGD is the ITERATIVE version of FGSM. Instead of one big step, it takes
    many small steps, projecting back into the ε-ball after each step.
    This makes it a strictly STRONGER attack than FGSM.

    Algorithm
    ─────────
    x_0 = x + uniform(-ε, ε)     ← random initialization within ε-ball

    For t = 1 to T:
        1. Compute ∂L/∂x_{t-1}              ← gradient of loss w.r.t. current image
        2. x_t = x_{t-1} + α·sign(∂L/∂x)   ← small step in gradient direction
        3. x_t = clip(x_t, x-ε, x+ε)        ← PROJECT back into ε-ball
        4. x_t = clip(x_t, -1, 1)            ← valid pixel range

    Why is PGD stronger?
    ────────────────────
    FGSM takes one big step of size ε. If the loss landscape is complex,
    this single step might not reach the optimal perturbation. PGD takes
    T steps of size α (where α < ε), effectively doing gradient ASCENT
    with projection — a constrained optimization that finds better attacks.

    Think of it like hiking:
    - FGSM: take one big blind leap uphill
    - PGD:  take many small steps uphill, checking the map after each one

    Args:
        model:           Frozen FaceNet model
        original_tensor: Input image tensor, shape (1, 3, 160, 160)
        epsilon:         L∞ bound (total max perturbation per pixel)
        alpha:           Step size per iteration (should be < ε/num_steps)
        num_steps:       Number of PGD iterations
        progress_callback: Optional callable(step, total) for progress reporting

    Returns:
        adversarial_tensor: Perturbed image
        noise:              The added perturbation
        cosine_sim:         Final cosine similarity (original vs adversarial)
    """
    model.eval()

    # ── Step 0: Get the ORIGINAL embedding (fixed reference) ─────────────
    with torch.no_grad():
        e_orig = model(original_tensor).detach()

    # ── Step 1: Random initialization within the ε-ball ──────────────────
    # Start from a random point in the ε-neighborhood, not from the original.
    # This helps escape local optima in the loss landscape.
    x_adv = original_tensor.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, -1.0, 1.0)
    x_adv = torch.max(torch.min(x_adv, original_tensor + epsilon), original_tensor - epsilon)

    # ── Step 2: Iterative gradient ascent with projection ────────────────
    for step in range(num_steps):
        # Enable gradient tracking on the current adversarial image
        x_adv = x_adv.detach().requires_grad_(True)

        # Forward pass
        e_adv = model(x_adv)

        # Loss: MSE between embeddings (we want to MAXIMIZE this)
        loss = nn.functional.mse_loss(e_adv, e_orig)

        # Compute ∂L/∂x via torch.autograd.grad (same as FGSM)
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=x_adv,
            create_graph=False,
        )[0]

        # Gradient ascent step (note the + sign, not -)
        # α · sign(∂L/∂x) — small step in the direction that increases loss
        x_adv = x_adv + alpha * grad.sign()

        # ── PROJECTION STEP ──────────────────────────────────────────────
        # This is what distinguishes PGD from iterative FGSM.
        # After each step, we project back onto the feasible set:
        #   1. L∞ ball: |x_adv - x_orig| ≤ ε  for each pixel
        #   2. Valid pixels: -1 ≤ x_adv ≤ 1
        x_adv = torch.max(torch.min(x_adv, original_tensor + epsilon), original_tensor - epsilon)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)

        # Report progress if callback provided
        if progress_callback is not None:
            progress_callback(step + 1, num_steps)

    # Restore original gradient state
    # ── Step 3: Final verification ───────────────────────────────────────
    noise = x_adv - original_tensor.detach()
    with torch.no_grad():
        e_adv_final = model(x_adv)
        cosine_sim = nn.functional.cosine_similarity(e_orig, e_adv_final).item()

    return x_adv, noise, cosine_sim
