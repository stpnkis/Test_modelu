#!/usr/bin/env python3
"""
AnomalyDINO — Benchmark adapter.

Unified entry point invoked by the orchestrator inside the Docker container.

    python src/run_benchmark.py \\
        --dataset-id casting --seed 42 --n-train 100 \\
        --preprocessing-mode baseline
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# ── Ensure shared and src are importable ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from shared.seed_utils import set_seed
from shared.preprocessing import (
    build_transforms,
    get_image_size,
    validate_transform_pipeline,
    log_transform_pipeline,
)
from shared.thresholding import compute_threshold
from shared.metrics import compute_image_metrics, compute_aupro
from shared.runtime_profiler import profile_model
from shared.results_io import save_results
from shared.split_manager import load_split_manifest, get_split_dir
from shared.dataset_schema import IMAGE_EXTENSIONS

from train import DINOv2FeatureExtractor, BACKBONE_CONFIG

logger = logging.getLogger(__name__)


def _get_image_paths(directory: Path) -> List[Path]:
    """Sorted list of image files in a directory."""
    return sorted(
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


@torch.no_grad()
def _score_single_image(
    image_path: Path,
    extractor: DINOv2FeatureExtractor,
    ref_norm: torch.Tensor,
    transform,
    device: torch.device,
) -> float:
    """Anomaly score for one image: max cosine distance to memory bank."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    patch_feats = extractor(tensor)
    test_patches = patch_feats.reshape(-1, patch_feats.shape[-1])
    test_norm = F.normalize(test_patches, dim=1)
    similarity = torch.mm(test_norm, ref_norm.t())
    max_sim, _ = similarity.max(dim=1)
    distances = 1.0 - max_sim
    return float(distances.max().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="AnomalyDINO benchmark run")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--preprocessing-mode", default="baseline")
    parser.add_argument(
        "--backbone", default="dinov2_vitb14", choices=sorted(BACKBONE_CONFIG)
    )
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

    # ── Validate threshold config ────────────────────────────────────
    logger.info(
        "Threshold config: strategy=%s  quantile_p=%s",
        args.threshold_strategy,
        args.threshold_quantile_p,
    )
    assert args.threshold_strategy in (
        "quantile",
        "max",
        "k_sigma",
    ), f"Invalid threshold strategy received: {args.threshold_strategy}"

    # ── Load split manifest ──────────────────────────────────────────
    split_dir = get_split_dir(
        args.splits_root, args.dataset_id, args.seed, args.n_train
    )
    manifest = load_split_manifest(
        args.splits_root, args.dataset_id, args.seed, args.n_train
    )
    logger.info("Split: %s", split_dir)

    # ── Paths ────────────────────────────────────────────────────────
    train_ok_dir = split_dir / "train" / "ok"
    val_ok_dir = split_dir / "val" / "ok"
    test_ok_dir = split_dir / "test" / "ok"
    test_nok_dir = split_dir / "test" / "nok"

    # ── Shared preprocessing ─────────────────────────────────────────
    transform = build_transforms(args.preprocessing_mode)
    validate_transform_pipeline(transform)
    log_transform_pipeline(transform, "anomalydino")

    # ── Build memory bank (training) ─────────────────────────────────
    logger.info("Building AnomalyDINO memory bank ...")
    fit_start = time.perf_counter()

    extractor = DINOv2FeatureExtractor(args.backbone).to(device)
    train_images = _get_image_paths(train_ok_dir)

    all_features = []
    for img_path in train_images:
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        feats = extractor(tensor)  # [1, h, w, D]
        all_features.append(feats.reshape(-1, feats.shape[-1]).cpu())

    memory_bank = torch.cat(all_features, dim=0).to(device)
    ref_norm = F.normalize(memory_bank, dim=1)

    fit_time = time.perf_counter() - fit_start
    logger.info(
        "Memory bank: %d patches × %d-dim  (%.1f s)",
        memory_bank.shape[0],
        memory_bank.shape[1],
        fit_time,
    )

    # ── Score val/ok for threshold ───────────────────────────────────
    logger.info("Scoring val/ok for threshold ...")
    val_images = _get_image_paths(val_ok_dir)
    val_scores = []
    for img_path in val_images:
        score = _score_single_image(img_path, extractor, ref_norm, transform, device)
        val_scores.append(score)

    threshold = compute_threshold(
        val_scores,
        strategy=args.threshold_strategy,
        quantile_p=args.threshold_quantile_p,
    )

    # ── Score test set ───────────────────────────────────────────────
    logger.info("Scoring test set ...")
    test_scores: List[float] = []
    test_labels: List[int] = []

    for img_path in _get_image_paths(test_ok_dir):
        score = _score_single_image(img_path, extractor, ref_norm, transform, device)
        test_scores.append(score)
        test_labels.append(0)

    for img_path in _get_image_paths(test_nok_dir):
        score = _score_single_image(img_path, extractor, ref_norm, transform, device)
        test_scores.append(score)
        test_labels.append(1)

    # ── Metrics ──────────────────────────────────────────────────────
    metrics = compute_image_metrics(test_labels, test_scores, threshold)

    # AU-PRO: AnomalyDINO does image-level scoring only (no anomaly map)
    aupro_result = compute_aupro(None, None)
    metrics.update(aupro_result)

    logger.info("Metrics: %s", metrics)

    # ── Runtime profiling ────────────────────────────────────────────
    # Pick a representative test image for profiling
    test_image_path = (
        _get_image_paths(test_ok_dir)[0]
        if _get_image_paths(test_ok_dir)
        else _get_image_paths(test_nok_dir)[0]
    )
    from PIL import Image as PILImage

    sample_img = PILImage.open(test_image_path).convert("RGB")
    sample_tensor = transform(sample_img).unsqueeze(0).to(device)

    def model_only_fn():
        feats = extractor(sample_tensor)
        patches = feats.reshape(-1, feats.shape[-1])
        t_norm = F.normalize(patches, dim=1)
        sim = torch.mm(t_norm, ref_norm.t())
        _ = sim.max(dim=1)

    def end_to_end_fn():
        _score_single_image(test_image_path, extractor, ref_norm, transform, device)

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

    # ── Save results ─────────────────────────────────────────────────
    save_results(
        experiments_root=args.experiments_root,
        dataset_id=args.dataset_id,
        model_name="anomalydino",
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
        },
        preprocessing_mode=args.preprocessing_mode,
        n_train=args.n_train,
    )

    logger.info("Done. AUROC=%.4f  F1=%.4f", metrics["auroc"], metrics["f1"])


if __name__ == "__main__":
    main()
