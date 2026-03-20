"""
Preprocessing — unified transform builder for all models.

This module is the **single source of truth** for image transforms.
No model is permitted to apply additional normalization or resizing.

Modes:
    ``baseline``      — Resize(256) → CenterCrop(224) → Normalize (default)
    ``high_accuracy`` — Resize(448) → Normalize
    ``edge_safe``     — Resize+Pad to square → Normalize (no crop)

Masks are transformed with the same geometry but nearest interpolation.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ── ImageNet constants ───────────────────────────────────────────────────────

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# ── Mode configurations ─────────────────────────────────────────────────────

PREPROCESSING_MODES = {
    "baseline": {
        "resize_shortest_edge": 256,
        "center_crop": 224,
        "image_size": 224,
    },
    "high_accuracy": {
        "resize_square": 448,
        "center_crop": None,
        "image_size": 448,
    },
    "edge_safe": {
        "keep_aspect_ratio": True,
        "pad_to_square": True,
        "target_size": 224,
        "center_crop": None,
        "image_size": 224,
    },
}


def get_image_size(mode: str = "baseline") -> int:
    """Return the final image size for the given mode."""
    if mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"Unknown preprocessing mode '{mode}'. "
            f"Available: {sorted(PREPROCESSING_MODES)}"
        )
    return PREPROCESSING_MODES[mode]["image_size"]


# ── Edge-safe helpers ────────────────────────────────────────────────────────


class ResizeKeepAspectAndPad:
    """Resize so longest edge equals *target_size*, then pad to square."""

    def __init__(self, target_size: int, fill: int = 0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: torch.Tensor | "PIL.Image.Image"):
        # PIL path
        from PIL import Image

        if isinstance(img, Image.Image):
            w, h = img.size
            scale = self.target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = TF.resize(img, [new_h, new_w])
            pad_w = self.target_size - new_w
            pad_h = self.target_size - new_h
            padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
            img = TF.pad(img, padding, fill=self.fill)
            return img
        # Tensor path
        _, h, w = img.shape
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, [new_h, new_w])
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        img = TF.pad(img, padding, fill=self.fill)
        return img


class MaskResizeKeepAspectAndPad:
    """Same geometry as ``ResizeKeepAspectAndPad`` but nearest interpolation."""

    def __init__(self, target_size: int, fill: int = 0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        from PIL import Image

        if isinstance(img, Image.Image):
            w, h = img.size
            scale = self.target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.NEAREST)
            pad_w = self.target_size - new_w
            pad_h = self.target_size - new_h
            padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
            img = TF.pad(img, padding, fill=self.fill)
            return img
        _, h, w = img.shape
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, [new_h, new_w], interpolation=T.InterpolationMode.NEAREST)
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        img = TF.pad(img, padding, fill=self.fill)
        return img


# ── Public API ───────────────────────────────────────────────────────────────


def build_transforms(mode: str = "baseline") -> T.Compose:
    """Build the image transform pipeline for the given mode.

    All modes:
      - Convert to RGB
      - Apply geometric transforms (mode-dependent)
      - ToTensor
      - ImageNet Normalize
      - No augmentations (fully deterministic)

    Args:
        mode: One of ``"baseline"``, ``"high_accuracy"``, ``"edge_safe"``.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    if mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"Unknown preprocessing mode '{mode}'. "
            f"Available: {sorted(PREPROCESSING_MODES)}"
        )

    steps: list = [T.Lambda(lambda img: img.convert("RGB"))]

    if mode == "baseline":
        steps.append(T.Resize(256, interpolation=T.InterpolationMode.BILINEAR))
        steps.append(T.CenterCrop(224))
    elif mode == "high_accuracy":
        steps.append(T.Resize((448, 448), interpolation=T.InterpolationMode.BILINEAR))
    elif mode == "edge_safe":
        steps.append(ResizeKeepAspectAndPad(224))

    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(steps)


def build_mask_transforms(mode: str = "baseline") -> T.Compose:
    """Build the mask transform pipeline matching the image geometry.

    Uses **nearest** interpolation for all resize/crop operations.
    Outputs a float tensor in [0, 1].
    """
    if mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"Unknown preprocessing mode '{mode}'. "
            f"Available: {sorted(PREPROCESSING_MODES)}"
        )

    steps: list = [T.Grayscale(num_output_channels=1)]

    if mode == "baseline":
        steps.append(T.Resize(256, interpolation=T.InterpolationMode.NEAREST))
        steps.append(T.CenterCrop(224))
    elif mode == "high_accuracy":
        steps.append(T.Resize((448, 448), interpolation=T.InterpolationMode.NEAREST))
    elif mode == "edge_safe":
        steps.append(MaskResizeKeepAspectAndPad(224))

    steps.append(T.ToTensor())
    return T.Compose(steps)


# ── Pipeline validation ──────────────────────────────────────────────────────


def validate_transform_pipeline(pipeline: T.Compose) -> None:
    """Validate that a transform pipeline contains no duplicate operations.

    Raises ``RuntimeError`` if Normalize, Resize, or CenterCrop appear
    more than once — a sign of double preprocessing.
    """
    counts = {"Normalize": 0, "Resize": 0, "CenterCrop": 0}

    for t in pipeline.transforms:
        cls_name = type(t).__name__
        if cls_name in counts:
            counts[cls_name] += 1
        # Also check inside nested Compose
        if isinstance(t, T.Compose):
            for inner in t.transforms:
                inner_name = type(inner).__name__
                if inner_name in counts:
                    counts[inner_name] += 1

    duplicates = {k: v for k, v in counts.items() if v > 1}
    if duplicates:
        raise RuntimeError(
            f"Benchmark pipeline validation failed: duplicate transforms "
            f"detected: {duplicates}. This indicates double preprocessing. "
            f"All models must use shared/preprocessing.build_transforms() "
            f"as the single source of truth."
        )


def log_transform_pipeline(pipeline: T.Compose, model_name: str) -> None:
    """Log the effective transform pipeline for auditability."""
    import logging

    _logger = logging.getLogger(__name__)
    steps = [f"  {i}: {t}" for i, t in enumerate(pipeline.transforms)]
    _logger.info(
        "Effective transform pipeline for %s:\n%s", model_name, "\n".join(steps)
    )


def build_geometric_transforms(mode: str = "baseline") -> T.Compose:
    """Build ONLY the geometric part of the pipeline (no ToTensor, no Normalize).

    Useful for models like RD++ that need to inject noise between
    geometry and normalization.
    """
    if mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"Unknown preprocessing mode '{mode}'. "
            f"Available: {sorted(PREPROCESSING_MODES)}"
        )

    steps: list = [T.Lambda(lambda img: img.convert("RGB"))]

    if mode == "baseline":
        steps.append(T.Resize(256, interpolation=T.InterpolationMode.BILINEAR))
        steps.append(T.CenterCrop(224))
    elif mode == "high_accuracy":
        steps.append(T.Resize((448, 448), interpolation=T.InterpolationMode.BILINEAR))
    elif mode == "edge_safe":
        steps.append(ResizeKeepAspectAndPad(224))

    return T.Compose(steps)
