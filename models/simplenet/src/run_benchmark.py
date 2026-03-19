#!/usr/bin/env python3
"""
SimpleNet — Benchmark adapter.

Unified entry point invoked by the orchestrator inside the Docker container.
Uses shared module for preprocessing, thresholding, metrics, and profiling.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from shared.seed_utils import set_seed, worker_init_fn
from shared.preprocessing import build_transforms, get_image_size
from shared.thresholding import compute_threshold
from shared.metrics import compute_image_metrics, compute_aupro
from shared.runtime_profiler import profile_model
from shared.results_io import save_results
from shared.split_manager import get_split_dir, load_split_manifest
from shared.dataset_schema import IMAGE_EXTENSIONS

from train import (
    Adaptor,
    Discriminator,
    FeatureExtractor,
    train_simplenet,
)

logger = logging.getLogger(__name__)


# ── Dataset helper ───────────────────────────────────────────────────────────


class BenchmarkImageDataset(Dataset):
    """Load images from a directory with a shared transform."""

    def __init__(self, directory: Path, transform, label: int):
        self.items = sorted(
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.transform = transform
        self.label = label

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img = Image.open(self.items[idx]).convert("RGB")
        tensor = self.transform(img)
        return tensor, self.label, str(self.items[idx])


# ── Scoring helpers ──────────────────────────────────────────────────────────


@torch.no_grad()
def _score_batch(
    extractor: FeatureExtractor,
    adaptor: Adaptor,
    discriminator: Discriminator,
    images: torch.Tensor,
    device: torch.device,
) -> List[float]:
    """Score a batch of images. Returns list of image-level anomaly scores."""
    images = images.to(device)
    feats = extractor(images)
    B, C, H, W = feats.shape
    patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
    adapted = adaptor(patches)
    logits = discriminator(adapted)
    probs = torch.sigmoid(logits).reshape(B, H, W)
    image_scores = 0.7 * probs.amax(dim=(1, 2)) + 0.3 * probs.mean(dim=(1, 2))
    return image_scores.cpu().tolist()


def _score_directory(
    directory: Path,
    label: int,
    extractor: FeatureExtractor,
    adaptor: Adaptor,
    discriminator: Discriminator,
    transform,
    device: torch.device,
) -> tuple:
    """Score all images in a directory. Returns (scores, labels)."""
    ds = BenchmarkImageDataset(directory, transform, label)
    if len(ds) == 0:
        return [], []
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    scores = []
    labels = []
    for imgs, lbls, _ in loader:
        s = _score_batch(extractor, adaptor, discriminator, imgs, device)
        scores.extend(s)
        labels.extend(lbls.tolist())
    return scores, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleNet benchmark run")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--preprocessing-mode", default="baseline")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--splits-root", default="splits")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--threshold-strategy", default="quantile")
    parser.add_argument("--threshold-quantile-p", type=float, default=0.99)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = get_image_size(args.preprocessing_mode)

    # ── Load split ───────────────────────────────────────────────────
    split_dir = get_split_dir(
        args.splits_root, args.dataset_id, args.seed, args.n_train
    )
    manifest = load_split_manifest(
        args.splits_root, args.dataset_id, args.seed, args.n_train
    )

    train_ok_dir = split_dir / "train" / "ok"
    val_ok_dir = split_dir / "val" / "ok"
    test_ok_dir = split_dir / "test" / "ok"
    test_nok_dir = split_dir / "test" / "nok"

    transform = build_transforms(args.preprocessing_mode)

    # ── Train ────────────────────────────────────────────────────────
    logger.info("Training SimpleNet ...")
    fit_start = time.perf_counter()
    checkpoint_path = train_simplenet(
        split_dir=str(split_dir),
        output_dir=str(
            Path(args.experiments_root) / args.dataset_id / "simplenet" / str(args.seed)
        ),
        image_size=image_size,
        backbone=args.backbone,
        epochs=args.epochs,
    )
    fit_time = time.perf_counter() - fit_start

    # ── Load trained model ───────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    extractor = FeatureExtractor(cfg["backbone"], cfg["layers"]).to(device)
    adaptor = Adaptor(cfg["in_features"], cfg["adaptor_dim"]).to(device)
    discriminator = Discriminator(cfg["adaptor_dim"]).to(device)
    adaptor.load_state_dict(ckpt["adaptor_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    extractor.eval()
    adaptor.eval()
    discriminator.eval()

    # ── Score val/ok → threshold ─────────────────────────────────────
    logger.info("Scoring val/ok for threshold ...")
    val_scores, _ = _score_directory(
        val_ok_dir,
        0,
        extractor,
        adaptor,
        discriminator,
        transform,
        device,
    )
    threshold = compute_threshold(
        val_scores,
        strategy=args.threshold_strategy,
        quantile_p=args.threshold_quantile_p,
    )

    # ── Score test set ───────────────────────────────────────────────
    logger.info("Scoring test set ...")
    test_ok_scores, test_ok_labels = _score_directory(
        test_ok_dir,
        0,
        extractor,
        adaptor,
        discriminator,
        transform,
        device,
    )
    test_nok_scores, test_nok_labels = _score_directory(
        test_nok_dir,
        1,
        extractor,
        adaptor,
        discriminator,
        transform,
        device,
    )

    test_scores = test_ok_scores + test_nok_scores
    test_labels = test_ok_labels + test_nok_labels

    # ── Metrics ──────────────────────────────────────────────────────
    metrics = compute_image_metrics(test_labels, test_scores, threshold)
    aupro_result = compute_aupro(None, None)  # SimpleNet: no pixel-level anomaly map
    metrics.update(aupro_result)
    logger.info("Metrics: %s", metrics)

    # ── Runtime profiling ────────────────────────────────────────────
    sample_img = Image.open(sorted(test_ok_dir.iterdir())[0]).convert("RGB")
    sample_tensor = transform(sample_img).unsqueeze(0).to(device)

    def model_only_fn():
        feats = extractor(sample_tensor)
        B, C, H, W = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
        adapted = adaptor(patches)
        _ = discriminator(adapted)

    def end_to_end_fn():
        img = Image.open(sorted(test_ok_dir.iterdir())[0]).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        feats = extractor(t)
        B, C, H, W = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
        adapted = adaptor(patches)
        logits = discriminator(adapted)
        probs = torch.sigmoid(logits).reshape(B, H, W)
        _ = 0.7 * probs.amax(dim=(1, 2)) + 0.3 * probs.mean(dim=(1, 2))

    runtime = profile_model(
        model_only_fn=model_only_fn,
        end_to_end_fn=end_to_end_fn,
        device=device,
        preprocessing_mode=args.preprocessing_mode,
        image_size=image_size,
        fit_time_s=fit_time,
        config_snapshot={
            "backbone": args.backbone,
            "n_train": args.n_train,
            "seed": args.seed,
            "dataset_id": args.dataset_id,
        },
    )

    # ── Save ─────────────────────────────────────────────────────────
    save_results(
        experiments_root=args.experiments_root,
        dataset_id=args.dataset_id,
        model_name="simplenet",
        seed=args.seed,
        metrics=metrics,
        runtime=runtime.to_dict(),
        config={
            "backbone": args.backbone,
            "n_train": args.n_train,
            "preprocessing_mode": args.preprocessing_mode,
            "image_size": image_size,
            "threshold_strategy": args.threshold_strategy,
            "threshold_quantile_p": args.threshold_quantile_p,
            "epochs": args.epochs,
        },
    )

    logger.info("Done. AUROC=%.4f  F1=%.4f", metrics["auroc"], metrics["f1"])


if __name__ == "__main__":
    main()
