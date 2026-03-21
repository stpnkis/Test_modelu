"""
SimpleNet Training Script — Production
========================================

Trains a SimpleNet anomaly-detection model (CVPR 2023, Liu et al.) with
zero user interaction beyond providing a folder of OK images.

How SimpleNet works:
  1. A pretrained CNN backbone extracts patch-level features (frozen).
  2. A two-layer MLP adaptor maps features to a compact embedding space.
  3. Gaussian noise is added to adapted features to create synthetic anomalies.
  4. A lightweight discriminator (BCE) learns to separate normal vs. synthetic.
  5. At inference, per-patch discriminator scores are aggregated to image-level.

Architecture:
  - Backbone:        wide_resnet50_2 (pretrained ImageNet, frozen)
  - Feature layers:  layer2 (512ch) + layer3 (1024ch) → 1536ch concat
  - Adaptor:         Linear(1536→512) → BN → LeakyReLU(0.2) ×2
  - Discriminator:   Linear(512→256)  → BN → LeakyReLU(0.2) → Linear(256→1)

Training — critical implementation details that prevent collapse (ln(4)≈1.3863):
  1. Two SEPARATE optimizer.step() calls per iteration
  2. .detach() on adapted features before feeding to discriminator in Step 1
  3. BCE labels: real features → 0, noise-perturbed features → 1
  4. Noise injected into adapted feature space, NOT pixel space
  5. Anomaly score orientation: higher score = more anomalous

Optimizer setup (separate LRs recommended):
  - Adaptor:       Adam lr=2e-4, weight_decay=1e-5
  - Discriminator:  Adam lr=1e-4 (half of adaptor), weight_decay=1e-5
  - Rationale: The adaptor-discriminator pair has adversarial dynamics.
    If the discriminator learns too fast, the adaptor gets vanishing
    gradients. The 2:1 LR ratio keeps useful gradients for both.
  - CosineAnnealingLR on both for smooth decay.

Augmentation strategy for unknown products:
  Set n_augment_passes > 1 to enrich the feature bank with augmented
  images. Safe augmentations: flips, small rotation (±10°), mild
  color jitter.  These simulate normal variation (orientation, lighting)
  without creating artificial anomalies.
  Recommended: n_augment_passes=3 for production, =1 for experiments.

Checkpoint format (.pth):
  { adaptor_state_dict, discriminator_state_dict, config }
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# Common image file extensions (consistent with shared/dataset_schema)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Minimum number of OK training images for reliable results.
# Below this threshold, the feature bank is too small for the
# discriminator to learn a meaningful normal/anomaly boundary.
MIN_TRAIN_IMAGES = 50


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class ImageFolderSimple(Dataset):
    """Load images from a flat directory (no labels)."""

    def __init__(
        self, root: str | Path, transform: transforms.Compose | None = None
    ) -> None:
        self.paths: list[Path] = sorted(
            p
            for p in Path(root).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def make_transform(image_size: int) -> transforms.Compose:
    """Standard ImageNet-compatible preprocessing transform (deterministic)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_augment_transform(image_size: int) -> transforms.Compose:
    """Augmented transform for feature bank enrichment.

    Safe augmentations for industrial anomaly detection:
      - Horizontal/vertical flips  (parts viewed from any orientation)
      - Small rotation (±10°)      (alignment variation between captures)
      - Mild color jitter           (lighting variation)

    NOT included (would create artificial anomaly-like artifacts):
      - Cutout, random erasing, heavy crop, perspective distortion
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Feature Extractor (frozen backbone)
# ---------------------------------------------------------------------------


class FeatureExtractor(nn.Module):
    """Extract intermediate features from a pretrained backbone via hooks.

    For WideResNet-50-2 with input 256 × 256:
      - layer2 → [B, 512, 32, 32]
      - layer3 → [B, 1024, 16, 16]
    After bilinear upsampling of layer3 and concatenation → [B, 1536, 32, 32].
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.layers = layers or ["layer2", "layer3"]

        self.backbone = getattr(models, backbone_name)(weights="IMAGENET1K_V1")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._features: dict[str, torch.Tensor] = {}
        for name in self.layers:
            layer = dict(self.backbone.named_children())[name]
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        """Return a forward-hook that stores the layer output."""

        def hook(_module, _input, output):
            self._features[name] = output

        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate multi-scale features.

        Args:
            x: Input images [B, 3, H, W].

        Returns:
            Concatenated patch features [B, C_total, H_feat, W_feat].
        """
        self._features.clear()
        self.backbone(x)

        # Determine target spatial size (largest among selected layers)
        target_size: tuple[int, int] | None = None
        for name in self.layers:
            h, w = self._features[name].shape[2:]
            if target_size is None or h * w > target_size[0] * target_size[1]:
                target_size = (h, w)

        # Upsample all features to target size and concatenate
        aligned: list[torch.Tensor] = []
        for name in self.layers:
            feat = self._features[name]
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(feat)

        return torch.cat(aligned, dim=1)  # [B, C_total, H_feat, W_feat]


# ---------------------------------------------------------------------------
# SimpleNet components
# ---------------------------------------------------------------------------


class Adaptor(nn.Module):
    """Two-layer feature adaptor (matches SimpleNet paper architecture).

    Projects backbone patch features into a compact embedding space.
    BatchNorm after each linear layer controls feature scale so that
    a fixed noise_std produces consistent perturbations regardless
    of adaptor weight magnitude.
    """

    def __init__(self, in_dim: int, out_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    """Anomaly discriminator: outputs a raw logit per patch feature."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits [N, 1]


# ---------------------------------------------------------------------------
# Feature bank extraction
# ---------------------------------------------------------------------------


def _extract_features(
    extractor: FeatureExtractor,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Extract patch features from a dataloader.

    Returns:
        (features_tensor [N_patches, C], feat_h, feat_w)
    """
    all_features: list[torch.Tensor] = []
    feat_h = feat_w = 0

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            feats = extractor(images)
            B, C, H, W = feats.shape
            feat_h, feat_w = H, W
            patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
            all_features.append(patches.cpu())

    return torch.cat(all_features, 0), feat_h, feat_w


def build_feature_bank(
    train_dir: str | Path,
    extractor: FeatureExtractor,
    device: torch.device,
    image_size: int,
    batch_size: int,
    n_augment_passes: int = 1,
    benchmark_transform=None,
) -> tuple[torch.Tensor, int, int, int]:
    """Build the feature bank with optional augmentation passes.

    Args:
        train_dir:         Path to train/good/ directory.
        extractor:         Frozen FeatureExtractor.
        device:            Torch device.
        image_size:        Image resize target.
        batch_size:        Batch size for feature extraction.
        n_augment_passes:  Total passes (1 = base only, >1 = base + augmented).
                           Ignored in benchmark mode (benchmark_transform set).
        benchmark_transform: If set, use this transform exclusively (no local
                           transforms, no augmentation). For benchmark mode.

    Returns:
        (feature_bank, feat_h, feat_w, in_channels)
    """
    # --- Base pass (deterministic, no augmentation) ---
    if benchmark_transform is not None:
        base_transform = benchmark_transform
        # In benchmark mode, disable augmentation
        n_augment_passes = 1
    else:
        base_transform = make_transform(image_size)

    base_dataset = ImageFolderSimple(train_dir, transform=base_transform)
    n_workers = min(4, max(1, len(base_dataset)))

    base_loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
    )

    features, feat_h, feat_w = _extract_features(extractor, base_loader, device)
    all_banks = [features]
    logger.info(f"  Base pass: {features.shape[0]:,} patches (dim={features.shape[1]})")

    # --- Additional augmented passes ---
    if n_augment_passes > 1:
        aug_transform = make_augment_transform(image_size)
        for pass_idx in range(n_augment_passes - 1):
            aug_dataset = ImageFolderSimple(train_dir, transform=aug_transform)
            aug_loader = DataLoader(
                aug_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers,
                pin_memory=(device.type == "cuda"),
            )
            aug_features, _, _ = _extract_features(extractor, aug_loader, device)
            all_banks.append(aug_features)
            logger.info(
                f"  Augmented pass {pass_idx + 1}: {aug_features.shape[0]:,} patches"
            )

    feature_bank = torch.cat(all_banks, dim=0)
    in_channels = feature_bank.shape[1]
    return feature_bank, feat_h, feat_w, in_channels


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train_simplenet(
    split_dir: str,
    output_dir: str,
    image_size: int = 256,
    backbone: str = "wide_resnet50_2",
    batch_size: int = 32,
    epochs: int = 200,
    lr: float = 2e-4,
    disc_lr: float = 1e-4,
    noise_std: float = 0.015,
    adaptor_dim: int = 512,
    weight_decay: float = 1e-5,
    patience: int = 30,
    n_augment_passes: int = 1,
    min_train_images: int = 50,
    benchmark_transform=None,
) -> str:
    """
    Train SimpleNet on a prepared dataset split.

    Two-step training loop (CRITICAL for stability — prevents ln(4) collapse):

      Step 1 — Discriminator update:
        adapted = adaptor(features).detach()   ← .detach() is critical
        anomaly = adapted + gaussian_noise
        loss_d = BCE(disc(adapted), 0) + BCE(disc(anomaly), 1)
        Only discriminator weights are updated.

      Step 2 — Adaptor update:
        adapted_fresh = adaptor(features)      ← fresh pass, NO detach
        loss_a = BCE(disc(adapted_fresh), 0)
        Only adaptor weights are updated.

    The .detach() in Step 1 decouples the two optimisation targets:
    the discriminator learns the real/fake boundary on frozen features,
    while the adaptor learns to project normal features into the region
    the discriminator considers "normal".

    Args:
        split_dir:          Path to split (must contain train/good/).
        output_dir:         Where to save model checkpoints.
        image_size:         Resize target (square).
        backbone:           Torchvision backbone name.
        batch_size:         Batch size for backbone feature extraction.
        epochs:             Maximum training epochs.
        lr:                 Adaptor learning rate (Adam).
        disc_lr:            Discriminator learning rate (Adam), typically lr/2.
        noise_std:          Gaussian noise std for synthetic anomalies.
                            Applied to adapted features (after BN, so features
                            have ~unit scale; 0.015 is ~1.5% perturbation/dim).
        adaptor_dim:        Output dimension of the 2-layer adaptor MLP.
        weight_decay:       L2 regularization for both optimizers.
        patience:           Early stopping patience (epochs w/o improvement).
        n_augment_passes:   Feature extraction passes (1=base, >1=+augmented).
        min_train_images:   Minimum OK images required (ValueError if fewer).

    Returns:
        Path to the best model checkpoint (.pth file).
    """
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Training SimpleNet  |  split={split_dir}  backbone={backbone}  "
        f"device={device}"
    )

    # ------------------------------------------------------------------
    # Guard: minimum training images
    # ------------------------------------------------------------------
    # Support both new (ok/) and legacy (good/) directory names
    train_dir = split_dir / "train" / "ok"
    if not train_dir.is_dir():
        train_dir = split_dir / "train" / "good"
    n_images = sum(
        1
        for p in train_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if n_images < min_train_images:
        msg = (
            f"\n{'=' * 60}\n"
            f"  TRAINING REJECTED: not enough OK images\n"
            f"  Found:    {n_images} images\n"
            f"  Required: at least {min_train_images}\n"
            f"\n"
            f"  SimpleNet needs ≥ {min_train_images} OK images to build\n"
            f"  a reliable feature bank. With fewer images the model\n"
            f"  cannot learn a stable normal distribution and results\n"
            f"  will be unreliable.\n"
            f"\n"
            f"  Please collect more OK samples and try again.\n"
            f"{'=' * 60}"
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Training images: {n_images}")

    # ------------------------------------------------------------------
    # Step 1: Extract feature bank
    # ------------------------------------------------------------------
    layers = ["layer2", "layer3"]
    extractor = FeatureExtractor(backbone, layers).to(device)

    feature_bank, feat_h, feat_w, in_features = build_feature_bank(
        train_dir=train_dir,
        extractor=extractor,
        device=device,
        image_size=image_size,
        batch_size=batch_size,
        n_augment_passes=n_augment_passes,
        benchmark_transform=benchmark_transform,
    )

    logger.info(
        f"Feature bank: {feature_bank.shape[0]:,} patches, "
        f"dim={in_features}, spatial=({feat_h}×{feat_w})"
    )

    # Free backbone GPU memory (no longer needed during MLP training)
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 2: Build adaptor and discriminator
    # ------------------------------------------------------------------
    adaptor = Adaptor(in_features, adaptor_dim).to(device)
    discriminator = Discriminator(adaptor_dim).to(device)

    n_params = sum(p.numel() for p in adaptor.parameters()) + sum(
        p.numel() for p in discriminator.parameters()
    )
    logger.info(f"Trainable parameters: {n_params:,}")

    # Separate optimizers with different LRs (see module docstring).
    # Weight decay provides mild L2 regularization.
    optimizer_adapt = torch.optim.Adam(
        adaptor.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    optimizer_disc = torch.optim.Adam(
        discriminator.parameters(),
        lr=disc_lr,
        weight_decay=weight_decay,
    )

    # Cosine annealing: smooth LR decay avoids the "LR cliff" problem
    # and works well with early stopping.
    scheduler_adapt = CosineAnnealingLR(optimizer_adapt, T_max=epochs, eta_min=1e-6)
    scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=epochs, eta_min=1e-6)

    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Step 3: Train on feature bank (two-step per batch)
    # ------------------------------------------------------------------
    patch_bs = min(256, len(feature_bank))
    patch_dataset = TensorDataset(feature_bank)
    patch_loader = DataLoader(
        patch_dataset,
        batch_size=patch_bs,
        shuffle=True,
        drop_last=len(feature_bank) > patch_bs,
    )

    adaptor.train()
    discriminator.train()

    best_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_ckpt_path = output_dir / "model_best.pth"
    last_ckpt_path = output_dir / "model.pth"

    logger.info(f"Noise std: {noise_std:.4f}")
    logger.info(
        f"Training: max {epochs} epochs, early stopping patience={patience}, "
        f"patch_batch={patch_bs}, weight_decay={weight_decay}"
    )

    for epoch in range(1, epochs + 1):
        epoch_loss_d = 0.0
        epoch_loss_a = 0.0
        n_batches = 0

        for (batch_feats,) in patch_loader:
            batch_feats = batch_feats.to(device)

            # ============================================================
            # Step A: Update DISCRIMINATOR (adaptor frozen via .detach())
            # ============================================================
            # .detach() is CRITICAL here — without it the discriminator
            # loss would backprop into the adaptor, coupling the two
            # optimisation targets and causing training collapse.
            adapted = adaptor(batch_feats)
            adapted_detached = adapted.detach()

            # Noise injected into ADAPTED feature space (not pixel space).
            # After adaptor BatchNorm, features have ~unit scale, so
            # noise_std=0.015 ≈ 1.5% perturbation per dimension.
            noise = torch.randn_like(adapted_detached) * noise_std
            anomaly_feats = adapted_detached + noise

            d_normal = discriminator(adapted_detached)
            d_anomaly = discriminator(anomaly_feats)

            # BCE labels: real features → 0, noise-perturbed → 1
            loss_d = criterion(d_normal, torch.zeros_like(d_normal)) + criterion(
                d_anomaly, torch.ones_like(d_anomaly)
            )

            optimizer_disc.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_disc.step()

            # ============================================================
            # Step B: Update ADAPTOR (discriminator provides signal)
            # ============================================================
            # Fresh forward pass — NO detach.  The adaptor learns to
            # project normal features into the region the discriminator
            # scores as "normal" (label = 0).
            adapted_fresh = adaptor(batch_feats)
            a_normal = discriminator(adapted_fresh)

            loss_a = criterion(a_normal, torch.zeros_like(a_normal))

            optimizer_adapt.zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(adaptor.parameters(), max_norm=1.0)
            optimizer_adapt.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_a += loss_a.item()
            n_batches += 1

        avg_d = epoch_loss_d / max(n_batches, 1)
        avg_a = epoch_loss_a / max(n_batches, 1)
        total = avg_d + avg_a

        # LR scheduling (per epoch)
        scheduler_adapt.step()
        scheduler_disc.step()

        # --- Early stopping check ---
        min_delta = 1e-4
        if total < best_loss - min_delta:
            best_loss = total
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(
                best_ckpt_path,
                adaptor,
                discriminator,
                backbone,
                layers,
                adaptor_dim,
                noise_std,
                image_size,
                in_features,
                feat_h,
                feat_w,
            )
        else:
            patience_counter += 1

        # --- Logging ---
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            lr_a = scheduler_adapt.get_last_lr()[0]
            lr_d = scheduler_disc.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch:>3}/{epochs}  "
                f"loss_d={avg_d:.4f}  loss_a={avg_a:.4f}  "
                f"total={total:.4f}  best={best_loss:.4f}  "
                f"lr_a={lr_a:.2e}  lr_d={lr_d:.2e}  "
                f"patience={patience_counter}/{patience}"
            )

        # --- Early stopping trigger ---
        if patience_counter >= patience:
            logger.info(
                f"  Early stopping at epoch {epoch} "
                f"(best epoch: {best_epoch}, best loss: {best_loss:.4f})"
            )
            break

    # Save last-epoch checkpoint (for comparison / debugging)
    _save_checkpoint(
        last_ckpt_path,
        adaptor,
        discriminator,
        backbone,
        layers,
        adaptor_dim,
        noise_std,
        image_size,
        in_features,
        feat_h,
        feat_w,
    )

    logger.info(
        f"Best checkpoint: {best_ckpt_path}  "
        f"(epoch {best_epoch}, loss={best_loss:.4f})"
    )
    logger.info(f"Last checkpoint: {last_ckpt_path}")
    return str(best_ckpt_path)


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: Path,
    adaptor: Adaptor,
    discriminator: Discriminator,
    backbone: str,
    layers: list[str],
    adaptor_dim: int,
    noise_std: float,
    image_size: int,
    in_features: int,
    feat_h: int,
    feat_w: int,
) -> None:
    """Persist adaptor + discriminator weights and training config to disk."""
    config = {
        "backbone": backbone,
        "layers": layers,
        "adaptor_dim": adaptor_dim,
        "noise_std": noise_std,
        "image_size": image_size,
        "in_features": in_features,
        "feat_h": feat_h,
        "feat_w": feat_w,
    }
    torch.save(
        {
            "adaptor_state_dict": adaptor.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "config": config,
        },
        path,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train SimpleNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-dir", type=str, required=True, help="Path to dataset split directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/train_output",
        help="Path to save model checkpoint and logs",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Image batch size for feature extraction",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs (early stopping may end sooner)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Adaptor learning rate (Adam)"
    )
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=1e-4,
        help="Discriminator learning rate (Adam), typically lr/2",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.015,
        help="Gaussian noise std for synthetic anomalies "
        "(paper default: 0.015; try 0.01–0.05)",
    )
    parser.add_argument(
        "--adaptor-dim",
        type=int,
        default=512,
        help="Output dimension of the 2-layer adaptor MLP",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 regularization for both optimizers",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (epochs w/o improvement)",
    )
    parser.add_argument(
        "--n-augment-passes",
        type=int,
        default=1,
        help="Feature extraction passes (1=base, >1=base+augmented)",
    )
    parser.add_argument(
        "--min-train-images",
        type=int,
        default=50,
        help="Minimum OK images required to start training",
    )

    args = parser.parse_args()

    ckpt = train_simplenet(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        backbone=args.backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        disc_lr=args.disc_lr,
        noise_std=args.noise_std,
        adaptor_dim=args.adaptor_dim,
        weight_decay=args.weight_decay,
        patience=args.patience,
        n_augment_passes=args.n_augment_passes,
        min_train_images=args.min_train_images,
    )
    print(f"\nModel checkpoint: {ckpt}")
