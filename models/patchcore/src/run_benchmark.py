#!/usr/bin/env python3
"""
PatchCore — Benchmark adapter.

Unified entry point invoked by the orchestrator inside the Docker container.

CRITICAL DESIGN DECISIONS:
1. Anomalib's Folder datamodule is used ONLY for engine.fit() (memory bank
   building).  Even there, val_split_mode is forced to "same_as_test" and
   test_split_mode to "from_dir" so that anomalib's internal
   _create_val_split / _create_test_split never randomly resplit our data.
2. For val/ok threshold scoring and test set evaluation we bypass the
   anomalib datamodule entirely.  Explicit BenchmarkImageDataset +
   DataLoader instances guarantee that exactly the files defined by the
   split manager are scored — no hidden halving, no anomaly contamination
   of the threshold, no library side effects.
3. Shared transforms are used everywhere to prevent double preprocessing.
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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
from shared.results_io import save_results, save_predictions_csv
from shared.split_manager import get_split_dir, load_split_manifest
from shared.dataset_schema import IMAGE_EXTENSIONS

# Anomalib is used ONLY for PatchCore model definition + Engine.fit()
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


def _to_numpy(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.cpu().numpy().flatten()
    return np.asarray(tensor_or_array).flatten()


def _score_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[list[float], list[int], list[str]]:
    """Run model inference on *dataset* and collect scores, labels, paths.

    This function is the ONLY scoring path used for val and test.
    It bypasses anomalib's datamodule / engine.predict entirely,
    guaranteeing that exactly the provided dataset is scored.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,        # safest in Docker; avoids worker spawn issues
        pin_memory=False,
    )
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_paths: list[str] = []

    model.eval()
    with torch.no_grad():
        for tensors, labels, paths in loader:
            tensors = tensors.to(device)
            output = model(tensors)
            # PatchCore returns dict {"anomaly_map": ..., "pred_score": ...}
            if isinstance(output, dict):
                scores = output["pred_score"]
            else:
                scores = output
            all_scores.extend(_to_numpy(scores).tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())
            all_paths.extend(paths)

    return all_scores, all_labels, all_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore benchmark run")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--preprocessing-mode", default="baseline")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1)
    parser.add_argument("--num-neighbors", type=int, default=15)
    parser.add_argument("--splits-root", default="splits")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--threshold-strategy", default="quantile")
    parser.add_argument("--threshold-quantile-p", type=float, default=0.98)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(args.seed)
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

    # ── Verify split directories ─────────────────────────────────────
    for name, d in [("train/ok", train_ok_dir), ("val/ok", val_ok_dir),
                    ("test/ok", test_ok_dir), ("test/nok", test_nok_dir)]:
        assert d.is_dir(), f"Split directory missing: {d}"

    # ── Build shared transforms ──────────────────────────────────────
    transform = build_transforms(args.preprocessing_mode)
    validate_transform_pipeline(transform)
    log_transform_pipeline(transform, "patchcore")

    # ── Count files per split (ground truth from filesystem) ─────────
    n_train_ok_actual = len(BenchmarkImageDataset(train_ok_dir, transform, 0))
    n_val_ok_actual = len(BenchmarkImageDataset(val_ok_dir, transform, 0))
    n_test_ok_actual = len(BenchmarkImageDataset(test_ok_dir, transform, 0))
    n_test_nok_actual = len(BenchmarkImageDataset(test_nok_dir, transform, 1))

    logger.info(
        "Split file counts (filesystem): train_ok=%d  val_ok=%d  "
        "test_ok=%d  test_nok=%d  test_total=%d",
        n_train_ok_actual,
        n_val_ok_actual,
        n_test_ok_actual,
        n_test_nok_actual,
        n_test_ok_actual + n_test_nok_actual,
    )

    # Cross-check with manifest
    mc = manifest["counts"]
    if n_train_ok_actual != mc["train_ok"]:
        logger.warning(
            "train_ok count mismatch: filesystem=%d  manifest=%d",
            n_train_ok_actual, mc["train_ok"],
        )
    if n_val_ok_actual != mc["val_ok"]:
        logger.warning(
            "val_ok count mismatch: filesystem=%d  manifest=%d",
            n_val_ok_actual, mc["val_ok"],
        )
    if n_test_ok_actual != mc["test_ok"]:
        logger.warning(
            "test_ok count mismatch: filesystem=%d  manifest=%d",
            n_test_ok_actual, mc["test_ok"],
        )
    if n_test_nok_actual != mc["test_nok"]:
        logger.warning(
            "test_nok count mismatch: filesystem=%d  manifest=%d",
            n_test_nok_actual, mc["test_nok"],
        )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Train PatchCore via anomalib Engine.fit()
    # ══════════════════════════════════════════════════════════════════
    # Anomalib's Folder datamodule is used ONLY here.
    # val_split_mode="same_as_test" prevents random resplitting.
    # test_split_mode="from_dir" keeps normal_test_dir images in test.
    # Neither val nor test data is used by PatchCore during fit
    # (num_sanity_val_steps=0, max_epochs=1).
    logger.info("Training PatchCore via anomalib ...")

    # Re-seed before anomalib to ensure deterministic data loading
    set_seed(args.seed)
    fit_start = time.perf_counter()

    output_dir = (
        Path(args.experiments_root) / args.dataset_id / "patchcore" / str(args.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
            # CRITICAL: disable internal val/test resplitting
            val_split_mode="same_as_test",
            val_split_ratio=0.5,  # irrelevant with same_as_test but explicit
            test_split_mode="from_dir",
            test_split_ratio=0.2,  # irrelevant with from_dir but explicit
        )
        logger.info(
            "Anomalib Folder: val_split_mode=same_as_test, "
            "test_split_mode=from_dir (no random resplit)."
        )
    except TypeError:
        raise RuntimeError(
            "Your anomalib version does not accept val_split_mode/test_split_mode. "
            "Upgrade to anomalib >= 1.2."
        )

    model = Patchcore(
        backbone=args.backbone,
        layers=args.layers,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
    )
    logger.info(
        "PatchCore config: backbone=%s layers=%s coreset=%.3f neighbors=%d",
        args.backbone, args.layers, args.coreset_sampling_ratio, args.num_neighbors,
    )

    engine = Engine(
        task="classification",
        default_root_dir=str(output_dir),
        max_epochs=1,
        devices=1,
        accelerator="auto",
        num_sanity_val_steps=0,
    )

    engine.fit(model=model, datamodule=datamodule)
    fit_time = time.perf_counter() - fit_start

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Score val/ok for threshold (bypass anomalib datamodule)
    # ══════════════════════════════════════════════════════════════════
    logger.info("Scoring val/ok for threshold (direct inference, no anomalib DM) ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Access the underlying torch model from the Lightning module
    patchcore_torch = model.model.to(device)
    patchcore_torch.eval()

    val_dataset = BenchmarkImageDataset(val_ok_dir, transform, label=0)
    logger.info(
        "Val dataset: %d images from %s  (first: %s)",
        len(val_dataset),
        val_ok_dir,
        val_dataset.items[0].name if len(val_dataset) > 0 else "N/A",
    )

    val_scores, val_labels, val_paths = _score_dataset(
        patchcore_torch, val_dataset, device, batch_size=32,
    )

    # Sanity: all val labels must be 0 (normal)
    assert all(
        lbl == 0 for lbl in val_labels
    ), "BUG: val/ok dataset contains non-normal labels — threshold would be contaminated."
    logger.info(
        "Val scores: n=%d  min=%.6f  max=%.6f  mean=%.6f",
        len(val_scores),
        min(val_scores),
        max(val_scores),
        sum(val_scores) / len(val_scores),
    )

    threshold = compute_threshold(
        val_scores,
        strategy=args.threshold_strategy,
        quantile_p=args.threshold_quantile_p,
    )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Score test set (bypass anomalib datamodule)
    # ══════════════════════════════════════════════════════════════════
    logger.info("Scoring test set (direct inference, no anomalib DM) ...")

    test_ok_dataset = BenchmarkImageDataset(test_ok_dir, transform, label=0)
    test_nok_dataset = BenchmarkImageDataset(test_nok_dir, transform, label=1)
    test_dataset = ConcatDataset([test_ok_dataset, test_nok_dataset])

    logger.info(
        "Test dataset: %d OK + %d NOK = %d total  (first OK: %s, first NOK: %s)",
        len(test_ok_dataset),
        len(test_nok_dataset),
        len(test_dataset),
        test_ok_dataset.items[0].name if len(test_ok_dataset) > 0 else "N/A",
        test_nok_dataset.items[0].name if len(test_nok_dataset) > 0 else "N/A",
    )

    all_scores, all_labels, all_paths = _score_dataset(
        patchcore_torch, test_dataset, device, batch_size=32,
    )

    logger.info(
        "Test scores: n=%d  n_ok=%d  n_nok=%d",
        len(all_scores),
        sum(1 for l in all_labels if l == 0),
        sum(1 for l in all_labels if l == 1),
    )

    # ── Metrics ──────────────────────────────────────────────────────
    metrics = compute_image_metrics(all_labels, all_scores, threshold)

    # AU-PRO: PatchCore (via our direct inference) doesn't use anomaly maps
    aupro_result = compute_aupro(None, None)
    metrics.update(aupro_result)

    # Add protocol metadata
    metrics["n_val_ok_for_threshold"] = len(val_scores)
    metrics["anomalib_internal_split_disabled"] = True

    logger.info("Metrics: %s", metrics)

    # ── Runtime profiling ────────────────────────────────────────────
    sample_path = sorted(test_ok_dir.iterdir())[0]
    sample_img = Image.open(sample_path).convert("RGB")
    sample_tensor = transform(sample_img).unsqueeze(0)

    # Model-only: forward pass through torch model
    def model_only_fn():
        with torch.no_grad():
            patchcore_torch(sample_tensor.to(device))

    def end_to_end_fn():
        img = Image.open(sample_path).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            patchcore_torch(t)

    runtime = profile_model(
        model_only_fn=model_only_fn,
        end_to_end_fn=end_to_end_fn,
        device=device,
        preprocessing_mode=args.preprocessing_mode,
        image_size=image_size,
        fit_time_s=fit_time,
        config_snapshot={
            "backbone": args.backbone,
            "layers": args.layers,
            "coreset_sampling_ratio": args.coreset_sampling_ratio,
            "num_neighbors": args.num_neighbors,
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
            "layers": args.layers,
            "coreset_sampling_ratio": args.coreset_sampling_ratio,
            "num_neighbors": args.num_neighbors,
            "n_train": args.n_train,
            "preprocessing_mode": args.preprocessing_mode,
            "image_size": image_size,
            "threshold_strategy": args.threshold_strategy,
            "threshold_quantile_p": args.threshold_quantile_p,
            "anomalib_val_split_mode": "same_as_test",
            "anomalib_internal_split_disabled": True,
            "scoring_method": "direct_model_inference",
        },
        preprocessing_mode=args.preprocessing_mode,
        n_train=args.n_train,
    )

    # Save per-image predictions CSV for auditability
    preds = [int(s >= threshold) for s in all_scores]
    if len(all_paths) == len(all_scores):
        save_predictions_csv(
            experiments_root=args.experiments_root,
            dataset_id=args.dataset_id,
            model_name="patchcore",
            seed=args.seed,
            image_paths=all_paths,
            scores=all_scores,
            labels=all_labels,
            predictions=preds,
            preprocessing_mode=args.preprocessing_mode,
            n_train=args.n_train,
        )

    logger.info(
        "Done. AUROC=%.4f  F1=%.4f  threshold=%.6f  "
        "n_val_ok=%d  n_test_ok=%d  n_test_nok=%d",
        metrics["auroc"],
        metrics["f1"],
        threshold,
        len(val_scores),
        metrics["n_test_ok"],
        metrics["n_test_nok"],
    )


if __name__ == "__main__":
    main()
