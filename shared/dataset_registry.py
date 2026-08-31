"""
Dataset registry — per-dataset split policy and methodology targets.

Each dataset declares:
  - split_policy: "official_test" or "custom_holdout"
  - val_ok_count: how many OK images to reserve for threshold calibration
  - test_ok_count: how many OK images in the test set (official or held-out)
  - few_shot_levels: which n_train sizes to evaluate in few-shot mode
  - source_info: provenance for auditing

Datasets with ``split_policy="official_test"`` (MVTec, VisA) preserve the
official test set exactly. val/ok is carved from the official train/ok pool
only. The test split is NEVER recreated or reshuffled.

Datasets with ``split_policy="custom_holdout"`` (Kaggle Casting) create one
fixed hold-out test set, frozen across all models and seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable per-dataset methodology configuration."""

    dataset_id: str
    split_policy: str  # "official_test" | "custom_holdout"

    # OK pool sizes (from the raw dataset, for documentation/validation)
    official_train_ok: int
    official_test_ok: int
    total_nok: int

    # Main benchmark targets
    val_ok_count: int
    main_train_ok: int  # = official_train_ok - val_ok_count

    # Few-shot levels (must include main_train_ok as the "full" level)
    few_shot_levels: List[int] = field(default_factory=list)

    # For custom_holdout: how many OK images to put in the hold-out test
    custom_test_ok_count: int = 0

    # Metadata
    source: str = ""
    has_masks: bool = False

    # Subdirectory within datasets/<id>/ containing actual data
    # (e.g. "can" for datasets/can/can/)
    data_subdir: str = ""


# ── Registry ─────────────────────────────────────────────────────────────────

# Actual file counts from the datasets/ directory:
#   bottle:  229 ok (209 train + 20 test merged), 63 nok, 63 masks
#   wood:    266 ok (247 train + 19 test merged), 60 nok, 60 masks  [NOTE: not leather]
#   pcb1:    1004 ok, 100 nok, 100 masks
#   casting: 519 ok, 781 nok, 0 masks

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "bottle": DatasetConfig(
        dataset_id="bottle",
        split_policy="official_test",
        official_train_ok=209,
        official_test_ok=20,
        total_nok=63,
        val_ok_count=50,
        main_train_ok=159,  # 209 - 50
        few_shot_levels=[10, 25, 50, 100, 159],
        source="MVTec AD — Bottle",
        has_masks=True,
    ),
    "wood": DatasetConfig(
        dataset_id="wood",
        split_policy="official_test",
        official_train_ok=247,
        official_test_ok=19,
        total_nok=60,
        val_ok_count=50,
        main_train_ok=197,  # 247 - 50
        few_shot_levels=[10, 25, 50, 100, 197],
        source="MVTec AD — Wood",
        has_masks=True,
    ),
    "pcb1": DatasetConfig(
        dataset_id="pcb1",
        split_policy="official_test",
        official_train_ok=904,   # total ok (1004) - 100 for test_ok
        official_test_ok=100,    # we treat 100 ok as official test pool
        total_nok=100,
        val_ok_count=100,
        main_train_ok=804,  # 904 - 100
        few_shot_levels=[10, 25, 50, 100, 200, 804],
        source="VisA — PCB1 (one-class protocol)",
        has_masks=True,
    ),
    "casting": DatasetConfig(
        dataset_id="casting",
        split_policy="custom_holdout",
        official_train_ok=0,   # no official split
        official_test_ok=0,    # no official split
        total_nok=781,
        val_ok_count=50,
        main_train_ok=365,  # 519 - 104 (test_ok) - 50 (val_ok)
        custom_test_ok_count=104,
        few_shot_levels=[10, 25, 50, 100, 200, 365],
        source="Kaggle Casting Product (512×512)",
        has_masks=False,
    ),
    "can": DatasetConfig(
        dataset_id="can",
        split_policy="fixed_official",
        official_train_ok=412,   # train/good
        official_test_ok=72,     # test_public/good
        total_nok=90,            # test_public/bad
        val_ok_count=46,         # validation/good
        main_train_ok=412,       # all train/good used for training
        few_shot_levels=[10, 25, 50, 100, 200, 412],
        source="MVTec AD 2 — Can (test_pub only)",
        has_masks=True,
        data_subdir="can",       # datasets/can/can/
    ),
    "vial": DatasetConfig(
        dataset_id="vial",
        split_policy="fixed_official",
        official_train_ok=291,   # train/good
        official_test_ok=35,     # test_public/good
        total_nok=105,           # test_public/bad
        val_ok_count=41,         # validation/good
        main_train_ok=291,       # all train/good used for training
        few_shot_levels=[10, 25, 50, 100, 200, 291],
        source="MVTec AD 2 — Vial (test_pub only)",
        has_masks=True,
        data_subdir="vial",      # datasets/vial/vial/
    ),
}


def get_dataset_config(dataset_id: str) -> DatasetConfig:
    """Return the methodology config for a dataset.

    Raises:
        KeyError: if dataset is not registered.
    """
    if dataset_id not in DATASET_CONFIGS:
        raise KeyError(
            f"Dataset '{dataset_id}' is not registered. "
            f"Available: {sorted(DATASET_CONFIGS)}. "
            f"Add it to shared/dataset_registry.py before running."
        )
    return DATASET_CONFIGS[dataset_id]


def get_main_n_train(dataset_id: str) -> int:
    """Return the main benchmark n_train for a dataset."""
    return get_dataset_config(dataset_id).main_train_ok


def get_few_shot_levels(dataset_id: str) -> List[int]:
    """Return the nested few-shot levels for a dataset."""
    return list(get_dataset_config(dataset_id).few_shot_levels)


def validate_dataset_counts(
    dataset_id: str,
    actual_ok_count: int,
    actual_nok_count: int,
) -> None:
    """Validate that actual file counts match the registry.

    Raises:
        ValueError: if counts don't match expectations.
    """
    cfg = get_dataset_config(dataset_id)

    if cfg.split_policy == "official_test":
        expected_ok = cfg.official_train_ok + cfg.official_test_ok
    elif cfg.split_policy == "fixed_official":
        expected_ok = cfg.official_train_ok + cfg.val_ok_count + cfg.official_test_ok
    else:
        expected_ok = cfg.custom_test_ok_count + cfg.val_ok_count + cfg.main_train_ok

    if actual_ok_count != expected_ok:
        raise ValueError(
            f"Dataset '{dataset_id}': expected {expected_ok} OK images "
            f"(registry), found {actual_ok_count}. "
            f"Update dataset_registry.py if the dataset has changed."
        )

    if actual_nok_count != cfg.total_nok:
        raise ValueError(
            f"Dataset '{dataset_id}': expected {cfg.total_nok} NOK images "
            f"(registry), found {actual_nok_count}. "
            f"Update dataset_registry.py if the dataset has changed."
        )
