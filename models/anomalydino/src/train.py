"""
AnomalyDINO — Training Script (Memory-Bank Builder)
=====================================================

Builds a reference memory bank from N normal (OK) images using the
AnomalyDINO approach (Damm et al., WACV 2025).

How it works
------------
1. A frozen DINOv2 ViT backbone extracts dense patch-level features
   from every reference image.
2. All patch features are concatenated into a single memory bank —
   shape ``[N_images × n_patches, feature_dim]``.
3. At inference (evaluate.py) each test-image patch is compared to
   the memory bank via cosine distance.  The maximum distance across
   all patches becomes the image-level anomaly score.

No gradient updates are involved — "training" is a single forward pass
through the reference images, identical in concept to PatchCore.

Supported backbones
-------------------
==============  ============  ==========  ==========
Backbone        Hub name      Feat dim    Patch size
==============  ============  ==========  ==========
dinov2_vits14   ViT-S/14      384         14
dinov2_vitb14   ViT-B/14      768         14
==============  ============  ==========  ==========

Checkpoint format (``.pth``)::

    { "memory_bank": Tensor[N_patches, D], "config": dict }

Usage (standalone)::

    python src/train.py --split-dir splits/n50 --output-dir experiments/n50

Typically called automatically by ``experiment_runner.py``.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F  # noqa: N812 — standard alias
from PIL import Image
from torchvision import transforms

__all__ = [
    "BACKBONE_CONFIG",
    "DEFAULT_BACKBONE",
    "DEFAULT_IMAGE_SIZE",
    "IMAGE_EXTENSIONS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DINOv2FeatureExtractor",
    "build_memory_bank",
    "train_anomalydino",
]

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
"""File extensions recognised as images (lower-cased before comparison)."""

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

BACKBONE_CONFIG: Dict[str, dict] = {
    "dinov2_vits14": {"hub_name": "dinov2_vits14", "feat_dim": 384, "patch_size": 14},
    "dinov2_vitb14": {"hub_name": "dinov2_vitb14", "feat_dim": 768, "patch_size": 14},
}
"""Mapping of backbone name → torch.hub model id, feature dimension, patch size."""

DEFAULT_BACKBONE: str = "dinov2_vitb14"
"""Default backbone — ViT-B/14 gives the best accuracy in our experiments."""

DEFAULT_IMAGE_SIZE: int = 448
"""Default input image size (448 / 14 = 32 patches per side = 1 024 per image)."""


# ── DINOv2 Feature Extractor ────────────────────────────────────────────────


class DINOv2FeatureExtractor(torch.nn.Module):
    """Extract dense patch features from a frozen DINOv2 ViT.

    With *image_size* 448 and *patch_size* 14 the spatial grid is
    32 × 32 = 1 024 patches.  Each patch has *feat_dim* dimensions
    (384 for ViT-S, 768 for ViT-B).

    **Note — first instantiation downloads DINOv2 weights from torch.hub
    (≈ 330 MB for ViT-B).**  Subsequent runs reuse the cache
    (``~/.cache/torch/hub``).
    """

    def __init__(self, backbone: str = DEFAULT_BACKBONE) -> None:
        super().__init__()
        if backbone not in BACKBONE_CONFIG:
            raise ValueError(
                f"Unknown backbone '{backbone}'.  "
                f"Supported: {sorted(BACKBONE_CONFIG)}"
            )
        self.cfg = BACKBONE_CONFIG[backbone]
        self.feat_dim: int = self.cfg["feat_dim"]
        self.patch_size: int = self.cfg["patch_size"]

        logger.info("Loading DINOv2 backbone: %s  (dim=%d)", backbone, self.feat_dim)
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            self.cfg["hub_name"],
            pretrained=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features.

        Args:
            x: ``[B, 3, H, W]`` — batch of images.

        Returns:
            ``[B, h, w, D]`` — spatial grid of patch features.
        """
        B, _C, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]  # [B, N, D]
        return tokens.reshape(B, h, w, self.feat_dim)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_image_paths(directory: Path) -> List[Path]:
    """Return sorted paths of all image files in *directory*."""
    return sorted(
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _make_transform(image_size: int) -> transforms.Compose:
    """ImageNet-normalised resize transform compatible with DINOv2."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ── Memory-Bank Builder ─────────────────────────────────────────────────────


@torch.no_grad()
def build_memory_bank(
    train_dir: str | Path,
    extractor: DINOv2FeatureExtractor,
    device: torch.device,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[torch.Tensor, int, int]:
    """Build the reference memory bank from OK images.

    Extracts dense DINOv2 patch features from every reference image and
    concatenates them into one large tensor.

    Args:
        train_dir:  Path to ``train/good/`` containing reference images.
        extractor:  :class:`DINOv2FeatureExtractor` already on *device*.
        device:     Torch device (``cuda`` or ``cpu``).
        image_size: Resize target (must be divisible by ``patch_size``).

    Returns:
        ``(memory_bank, feat_h, feat_w)``

        *memory_bank* has shape ``[N_images × feat_h × feat_w, feat_dim]``
        and lives on **CPU** (moved to GPU only when needed for inference).

    Raises:
        FileNotFoundError: If no loadable images are found.
    """
    train_dir = Path(train_dir)
    transform = _make_transform(image_size)
    image_paths = _get_image_paths(train_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {train_dir}")

    logger.info("Building memory bank from %d reference images …", len(image_paths))

    all_features: List[torch.Tensor] = []
    feat_h = feat_w = 0
    skipped = 0

    for img_path in image_paths:
        # Robust against occasional corrupt / truncated images
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            logger.warning("Skipping corrupt / unreadable image: %s", img_path)
            skipped += 1
            continue

        tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
        patch_feats = extractor(tensor)  # [1, h, w, D]
        _, feat_h, feat_w, D = patch_feats.shape
        all_features.append(patch_feats.reshape(-1, D).cpu())  # [h*w, D]

    if not all_features:
        raise FileNotFoundError(
            f"All images in {train_dir} failed to load ({skipped} skipped)."
        )

    memory_bank = torch.cat(all_features, dim=0)  # [k*h*w, D]

    logger.info(
        "Memory bank: %d patches × %d-dim  (%d images, %d×%d grid%s)",
        memory_bank.shape[0],
        memory_bank.shape[1],
        len(all_features),
        feat_h,
        feat_w,
        f", {skipped} skipped" if skipped else "",
    )
    return memory_bank, feat_h, feat_w


# ── Train (build + save) ────────────────────────────────────────────────────


def train_anomalydino(
    split_dir: str,
    output_dir: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    backbone: str = DEFAULT_BACKBONE,
) -> str:
    """Build the memory bank and save as a ``.pth`` checkpoint.

    This is the main entry point called by ``experiment_runner.py``
    (or directly via CLI).

    Args:
        split_dir:   Split root (must contain ``train/good/``).
        output_dir:  Where to write ``model_best.pth``.
        image_size:  Square resize target (divisible by 14).
        backbone:    ``"dinov2_vits14"`` or ``"dinov2_vitb14"``.

    Returns:
        Absolute path to the saved checkpoint.

    Raises:
        ValueError:       If *image_size* is not divisible by the patch size.
        FileNotFoundError: If the training directory has no images.
    """
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Training AnomalyDINO  |  split=%s  backbone=%s  device=%s",
        split_dir,
        backbone,
        device,
    )

    cfg = BACKBONE_CONFIG[backbone]
    if image_size % cfg["patch_size"] != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by "
            f"patch_size ({cfg['patch_size']})"
        )

    # 1. Load backbone ────────────────────────────────────────────────────
    extractor = DINOv2FeatureExtractor(backbone).to(device)

    # 2. Build memory bank ────────────────────────────────────────────────
    t0 = time.perf_counter()
    # Support both new (ok/) and legacy (good/) directory names
    _train_dir = split_dir / "train" / "ok"
    if not _train_dir.is_dir():
        _train_dir = split_dir / "train" / "good"
    memory_bank, feat_h, feat_w = build_memory_bank(
        train_dir=_train_dir,
        extractor=extractor,
        device=device,
        image_size=image_size,
    )
    build_time = time.perf_counter() - t0
    logger.info("Build time: %.2f s", build_time)

    # 3. Save checkpoint ──────────────────────────────────────────────────
    n_images = memory_bank.shape[0] // (feat_h * feat_w)
    checkpoint = {
        "memory_bank": memory_bank,
        "config": {
            "backbone": backbone,
            "image_size": image_size,
            "feat_dim": cfg["feat_dim"],
            "patch_size": cfg["patch_size"],
            "feat_h": feat_h,
            "feat_w": feat_w,
            "n_reference_images": n_images,
            "build_time_s": round(build_time, 4),
        },
    }
    ckpt_path = output_dir / "model_best.pth"
    torch.save(checkpoint, ckpt_path)
    logger.info("Checkpoint saved → %s", ckpt_path)

    # 4. Free GPU memory so evaluation can start cleanly ──────────────────
    del extractor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return str(ckpt_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build AnomalyDINO memory bank from reference images.",
    )
    parser.add_argument(
        "--split-dir",
        required=True,
        help="Path to the dataset split (must contain train/good/)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/train_output",
        help="Where to save model_best.pth  (default: experiments/train_output)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Input image size, must be divisible by 14  (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        choices=sorted(BACKBONE_CONFIG),
        help=f"DINOv2 backbone variant  (default: {DEFAULT_BACKBONE})",
    )
    args = parser.parse_args()

    path = train_anomalydino(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        backbone=args.backbone,
    )
    print(f"\nCheckpoint: {path}")
