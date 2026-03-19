#!/usr/bin/env python3
"""
RD++ — Benchmark adapter.

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
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
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
    BottleneckNetwork,
    Decoder,
    Encoder,
    MultiProjectionLayer,
    compute_anomaly_map,
    train_rd_plus_plus,
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
        return tensor, self.label


# ── Scoring ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def _score_images(
    encoder: Encoder,
    bn: BottleneckNetwork,
    decoder: Decoder,
    proj_layer: MultiProjectionLayer,
    dataloader: DataLoader,
    device: torch.device,
    image_size: int,
) -> tuple:
    """Score all images in a dataloader. Returns (scores, labels, anomaly_maps)."""
    all_scores = []
    all_labels = []
    all_maps = []

    for images, labels in dataloader:
        images = images.to(device)
        inputs = encoder(images)
        features = proj_layer(inputs)
        outputs = decoder(bn(features))
        anomaly_map = compute_anomaly_map(inputs, outputs, image_size)
        anomaly_map_np = anomaly_map.cpu().numpy()

        for j in range(anomaly_map_np.shape[0]):
            smoothed = scipy_gaussian_filter(anomaly_map_np[j], sigma=4)
            score = float(smoothed.max())
            all_scores.append(score)
            all_maps.append(smoothed)

        all_labels.extend(labels.tolist())

    return all_scores, all_labels, all_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="RD++ benchmark run")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--preprocessing-mode", default="baseline")
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
    logger.info("Training RD++ ...")
    fit_start = time.perf_counter()
    checkpoint_path = train_rd_plus_plus(
        split_dir=str(split_dir),
        output_dir=str(
            Path(args.experiments_root)
            / args.dataset_id
            / "rd_plus_plus"
            / str(args.seed)
        ),
        image_size=image_size,
        epochs=args.epochs,
    )
    fit_time = time.perf_counter() - fit_start

    # ── Load trained model ───────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder = Encoder().to(device).eval()
    bn = BottleneckNetwork(width_per_group=cfg["width_per_group"]).to(device)
    decoder = Decoder(
        layers=cfg["decoder_layers"], width_per_group=cfg["width_per_group"]
    ).to(device)
    proj_layer = MultiProjectionLayer(base=cfg["proj_base"]).to(device)
    bn.load_state_dict(ckpt["bn_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    proj_layer.load_state_dict(ckpt["proj_state_dict"])
    bn.eval()
    decoder.eval()
    proj_layer.eval()

    # ── Score val/ok → threshold ─────────────────────────────────────
    logger.info("Scoring val/ok for threshold ...")
    val_ds = BenchmarkImageDataset(val_ok_dir, transform, label=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    val_scores, _, _ = _score_images(
        encoder, bn, decoder, proj_layer, val_loader, device, image_size
    )

    threshold = compute_threshold(
        val_scores,
        strategy=args.threshold_strategy,
        quantile_p=args.threshold_quantile_p,
    )

    # ── Score test set ───────────────────────────────────────────────
    logger.info("Scoring test set ...")
    test_ok_ds = BenchmarkImageDataset(test_ok_dir, transform, label=0)
    test_nok_ds = BenchmarkImageDataset(test_nok_dir, transform, label=1)

    test_ok_loader = DataLoader(test_ok_ds, batch_size=1, shuffle=False, num_workers=0)
    test_nok_loader = DataLoader(
        test_nok_ds, batch_size=1, shuffle=False, num_workers=0
    )

    ok_scores, ok_labels, _ = _score_images(
        encoder, bn, decoder, proj_layer, test_ok_loader, device, image_size
    )
    nok_scores, nok_labels, nok_maps = _score_images(
        encoder, bn, decoder, proj_layer, test_nok_loader, device, image_size
    )

    test_scores = ok_scores + nok_scores
    test_labels = ok_labels + nok_labels

    # ── Metrics ──────────────────────────────────────────────────────
    metrics = compute_image_metrics(test_labels, test_scores, threshold)

    # AU-PRO: RD++ produces anomaly maps — check if masks are available
    has_masks = manifest.get("has_masks", False)
    masks_dir = split_dir / "test" / "masks"
    if has_masks and masks_dir.is_dir() and nok_maps:
        from shared.preprocessing import build_mask_transforms

        mask_transform = build_mask_transforms(args.preprocessing_mode)
        gt_masks = []
        for nok_path in sorted(test_nok_dir.iterdir()):
            if nok_path.is_file() and nok_path.suffix.lower() in IMAGE_EXTENSIONS:
                mask_path = masks_dir / nok_path.name
                if mask_path.exists():
                    mask_img = Image.open(mask_path)
                    mask_tensor = mask_transform(mask_img)
                    gt_masks.append(mask_tensor.squeeze(0).numpy())
        if len(gt_masks) == len(nok_maps):
            aupro_result = compute_aupro(nok_maps, gt_masks)
        else:
            aupro_result = compute_aupro(None, None)
    else:
        aupro_result = compute_aupro(None, None)

    metrics.update(aupro_result)
    logger.info("Metrics: %s", metrics)

    # ── Runtime profiling ────────────────────────────────────────────
    sample_img = Image.open(sorted(test_ok_dir.iterdir())[0]).convert("RGB")
    sample_tensor = transform(sample_img).unsqueeze(0).to(device)

    def model_only_fn():
        inputs = encoder(sample_tensor)
        features = proj_layer(inputs)
        outputs = decoder(bn(features))
        _ = compute_anomaly_map(inputs, outputs, image_size)

    def end_to_end_fn():
        img = Image.open(sorted(test_ok_dir.iterdir())[0]).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        inputs = encoder(t)
        features = proj_layer(inputs)
        outputs = decoder(bn(features))
        am = compute_anomaly_map(inputs, outputs, image_size)
        am_np = am.cpu().numpy()
        _ = scipy_gaussian_filter(am_np[0], sigma=4).max()

    runtime = profile_model(
        model_only_fn=model_only_fn,
        end_to_end_fn=end_to_end_fn,
        device=device,
        preprocessing_mode=args.preprocessing_mode,
        image_size=image_size,
        fit_time_s=fit_time,
        config_snapshot={
            "n_train": args.n_train,
            "seed": args.seed,
            "dataset_id": args.dataset_id,
        },
    )

    # ── Save ─────────────────────────────────────────────────────────
    save_results(
        experiments_root=args.experiments_root,
        dataset_id=args.dataset_id,
        model_name="rd_plus_plus",
        seed=args.seed,
        metrics=metrics,
        runtime=runtime.to_dict(),
        config={
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
