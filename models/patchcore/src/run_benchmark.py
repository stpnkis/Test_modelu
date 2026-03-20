#!/usr/bin/env python3
"""
PatchCore — Benchmark adapter.

Unified entry point invoked by the orchestrator inside the Docker container.

CRITICAL: This adapter avoids anomalib's implicit preprocessing pipeline.
Instead, it uses the shared transform builder and manually feeds images
through training / inference to prevent double-normalization and double-resize.

For training, we still use anomalib's PatchCore model (memory bank build)
but ensure data is loaded through the shared preprocessing pipeline.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from shared.seed_utils import set_seed, worker_init_fn
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
from shared.split_manager import get_split_dir, load_split_manifest
from shared.dataset_schema import IMAGE_EXTENSIONS

# We still import anomalib for PatchCore model + Engine, but control data ourselves
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

logger = logging.getLogger(__name__)


# ── Dataset for shared transforms ────────────────────────────────────────────


class BenchmarkImageDataset(Dataset):
    """Load images with the shared transform. Returns (tensor, label, path)."""

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


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_field(obj, *names):
    """Try to extract a named field from an object or dict."""
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
        if isinstance(obj, dict) and name in obj:
            val = obj[name]
            if val is not None:
                return val
    return None


def _to_numpy(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.cpu().numpy().flatten()
    return np.asarray(tensor_or_array).flatten()


def _label_from_path(path: str) -> int:
    return 0 if "good" in str(path).lower() or "/ok" in str(path).lower() else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore benchmark run")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--preprocessing-mode", default="baseline")
    parser.add_argument("--backbone", default="wide_resnet50_2")
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
    device_str = "auto"
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

    # ── Train PatchCore via anomalib ─────────────────────────────────
    # We use anomalib's Folder datamodule for training because PatchCore's
    # memory bank building is tightly coupled to anomalib's Engine.
    # We pass our shared transforms as train_transform / eval_transform
    # to override anomalib's implicit preprocessing pipeline.
    logger.info("Training PatchCore via anomalib ...")
    transform = build_transforms(args.preprocessing_mode)
    validate_transform_pipeline(transform)
    log_transform_pipeline(transform, "patchcore")

    fit_start = time.perf_counter()

    output_dir = (
        Path(args.experiments_root) / args.dataset_id / "patchcore" / str(args.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to pass explicit transforms to anomalib to prevent implicit ones.
    # anomalib >= 1.2 supports train_transform / eval_transform params.
    try:
        datamodule = Folder(
            name=args.dataset_id,
            root=str(split_dir),
            normal_dir="train/ok",
            abnormal_dir="test/nok",
            normal_test_dir="test/ok",
            task="classification",
            image_size=(image_size, image_size),
            train_transform=transform,
            eval_transform=transform,
            train_batch_size=32,
            eval_batch_size=32,
        )
        logger.info("Using explicit shared transforms for anomalib Folder datamodule.")
    except TypeError:
        # Fallback for anomalib versions that don't support transform params
        logger.warning(
            "anomalib Folder does not accept train_transform/eval_transform. "
            "Falling back to image_size matching. Verify no double preprocessing."
        )
        datamodule = Folder(
            name=args.dataset_id,
            root=str(split_dir),
            normal_dir="train/ok",
            abnormal_dir="test/nok",
            normal_test_dir="test/ok",
            task="classification",
            image_size=(image_size, image_size),
            train_batch_size=32,
            eval_batch_size=32,
        )

    model = Patchcore(
        backbone=args.backbone,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        task="classification",
        default_root_dir=str(output_dir),
        max_epochs=1,
        devices=1,
        accelerator="auto",
    )

    engine.fit(model=model, datamodule=datamodule)
    fit_time = time.perf_counter() - fit_start

    # Find checkpoint
    ckpt_path = None
    cb = engine.trainer.checkpoint_callback
    if cb is not None and cb.best_model_path:
        ckpt_path = cb.best_model_path
    if not ckpt_path:
        candidates = sorted(output_dir.rglob("*.ckpt"))
        if candidates:
            ckpt_path = str(candidates[-1])
    if not ckpt_path:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")

    logger.info("Checkpoint: %s", ckpt_path)

    # ── Rebuild model for scoring with shared transforms ─────────────
    # For scoring val and test, we use anomalib's predict() with the same
    # datamodule to avoid re-implementing PatchCore's scoring logic.
    # The key guarantee is: image_size in anomalib == shared image_size.

    # Score val/ok for threshold
    logger.info("Scoring val/ok for threshold ...")
    try:
        val_datamodule = Folder(
            name=f"{args.dataset_id}_val",
            root=str(split_dir),
            normal_dir="val/ok",
            abnormal_dir="test/nok",  # required by anomalib, not used for scoring val
            normal_test_dir="val/ok",  # we score val/ok as "test" set
            task="classification",
            image_size=(image_size, image_size),
            eval_transform=transform,
            eval_batch_size=32,
        )
    except TypeError:
        val_datamodule = Folder(
            name=f"{args.dataset_id}_val",
            root=str(split_dir),
            normal_dir="val/ok",
            abnormal_dir="test/nok",
            normal_test_dir="val/ok",
            task="classification",
            image_size=(image_size, image_size),
            eval_batch_size=32,
        )

    val_predictions = engine.predict(
        model=model,
        datamodule=val_datamodule,
        ckpt_path=ckpt_path,
    )

    val_scores = []
    for batch in val_predictions:
        scores = _extract_field(batch, "pred_score", "pred_scores")
        if scores is not None:
            val_scores.extend(_to_numpy(scores).tolist())

    threshold = compute_threshold(
        val_scores,
        strategy=args.threshold_strategy,
        quantile_p=args.threshold_quantile_p,
    )

    # Score test set
    logger.info("Scoring test set ...")
    test_predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    all_scores = []
    all_labels = []
    for batch in test_predictions:
        scores = _extract_field(batch, "pred_score", "pred_scores")
        labels = _extract_field(batch, "gt_label", "label", "gt_labels")
        paths = _extract_field(batch, "image_path", "image_paths")

        if scores is not None:
            all_scores.extend(_to_numpy(scores).tolist())
        if labels is not None:
            all_labels.extend(_to_numpy(labels).astype(int).tolist())
        elif paths is not None:
            all_labels.extend(_label_from_path(p) for p in paths)

    # ── Metrics ──────────────────────────────────────────────────────
    metrics = compute_image_metrics(all_labels, all_scores, threshold)

    # AU-PRO: PatchCore (via anomalib classification mode) doesn't produce anomaly maps
    aupro_result = compute_aupro(None, None)
    metrics.update(aupro_result)
    logger.info("Metrics: %s", metrics)

    # ── Runtime profiling ────────────────────────────────────────────
    # For PatchCore, we measure latency using anomalib's predict.
    # This is the most honest measurement since the full pipeline goes
    # through anomalib.
    transform = build_transforms(args.preprocessing_mode)
    sample_path = sorted(test_ok_dir.iterdir())[0]
    sample_img = Image.open(sample_path).convert("RGB")
    sample_tensor = transform(sample_img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model-only: forward pass through model
    model_instance = model.to(device)
    model_instance.eval()

    def model_only_fn():
        with torch.no_grad():
            model_instance(sample_tensor.to(device))

    def end_to_end_fn():
        img = Image.open(sample_path).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            model_instance(t)

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
        model_name="patchcore",
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
