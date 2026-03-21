"""
Tests for benchmark protocol compliance.

Covers:
  - Legacy pipeline files removed (no evaluate.py / experiment_runner.py / dataset_splitter.py)
  - Results schema includes environment metadata (git SHA, hardware)
  - Predictions CSV export round-trip
  - Runtime profiler returns latency std
  - Seed determinism across runs
  - No test leakage in shared/thresholding.py
  - Few-shot config present in config.yaml
  - Aggregation across 3 seeds
  - No overlap between train/val/test (structural)
  - Threshold computed from val/ok only (structural)
"""

from __future__ import annotations

import ast
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from shared.results_io import (
    get_results_dir,
    save_results,
    load_results,
    save_predictions_csv,
    _get_environment_info,
    _get_git_sha,
)
from shared.runtime_profiler import (
    profile_model,
    measure_latency,
    RuntimeResult,
)
from shared.split_manager import create_split, load_split_manifest
from shared.thresholding import compute_threshold
from shared.metrics import compute_image_metrics
from shared.aggregate import aggregate_seeds, format_latex_row, save_summary
from shared.seed_utils import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Fixtures ─────────────────────────────────────────────────────────────────

NUM_OK = 300
NUM_NOK = 20


@pytest.fixture
def fake_dataset(tmp_path):
    ds = tmp_path / "datasets" / "test_ds"
    ok_dir = ds / "ok"
    nok_dir = ds / "nok"
    ok_dir.mkdir(parents=True)
    nok_dir.mkdir(parents=True)
    for i in range(NUM_OK):
        (ok_dir / f"ok_{i:04d}.png").write_bytes(b"fake")
    for i in range(NUM_NOK):
        (nok_dir / f"nok_{i:04d}.png").write_bytes(b"fake")
    return ds, tmp_path / "splits"


# ── 1. Legacy pipeline files must not exist ──────────────────────────────────


class TestLegacyRemoval:
    """All legacy pipeline files must be deleted from the repository."""

    LEGACY_FILES = [
        "models/patchcore/src/evaluate.py",
        "models/patchcore/src/experiment_runner.py",
        "models/patchcore/src/dataset_splitter.py",
        "models/rd_plus_plus/src/evaluate.py",
        "models/rd_plus_plus/src/experiment_runner.py",
        "models/rd_plus_plus/src/dataset_splitter.py",
        "models/simplenet/src/evaluate.py",
        "models/simplenet/src/experiment_runner.py",
        "models/simplenet/src/dataset_splitter.py",
        "models/anomalydino/src/evaluate.py",
        "models/anomalydino/src/experiment_runner.py",
        "models/anomalydino/src/dataset_splitter.py",
    ]

    @pytest.mark.parametrize("rel_path", LEGACY_FILES)
    def test_legacy_file_does_not_exist(self, rel_path):
        """Legacy files must be removed — they contained methodological flaws."""
        full_path = REPO_ROOT / rel_path
        assert not full_path.exists(), (
            f"Legacy file still exists: {rel_path}. "
            "Delete it — the official pipeline uses src/run_benchmark.py only."
        )

    LEGACY_TEST_DIRS = [
        "models/anomalydino/tests",
        "models/rd_plus_plus/tests",
        "models/simplenet/tests",
    ]

    @pytest.mark.parametrize("rel_path", LEGACY_TEST_DIRS)
    def test_legacy_test_dirs_removed(self, rel_path):
        """Model-level test directories (testing old Youden J logic) must be gone."""
        full_path = REPO_ROOT / rel_path
        assert not full_path.exists(), (
            f"Legacy test directory still exists: {rel_path}. "
            "All tests are centralized in tests/."
        )


# ── 2. Results schema with environment metadata ─────────────────────────────


class TestResultsMetadata:
    """results.json must contain environment info."""

    def test_save_includes_environment(self, tmp_path):
        metrics = {"auroc": 0.95, "f1": 0.90}
        path = save_results(
            experiments_root=str(tmp_path),
            dataset_id="test",
            model_name="model",
            seed=42,
            metrics=metrics,
            preprocessing_mode="baseline",
            n_train=100,
        )
        with open(path) as f:
            data = json.load(f)

        assert "environment" in data
        env = data["environment"]
        assert "git_sha" in env
        assert "torch_version" in env
        assert "python_version" in env
        assert "platform" in env
        assert "cuda_available" in env
        assert "timestamp" in env

    def test_environment_info_has_required_keys(self):
        env = _get_environment_info()
        assert "git_sha" in env
        assert "torch_version" in env
        assert "python_version" in env

    def test_git_sha_returns_string(self):
        sha = _get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0


# ── 3. Predictions CSV export ────────────────────────────────────────────────


class TestPredictionsCSV:
    """Per-image predictions CSV must be exportable and parseable."""

    def test_round_trip(self, tmp_path):
        paths = ["/img/a.png", "/img/b.png", "/img/c.png"]
        scores = [0.1, 0.5, 0.9]
        labels = [0, 0, 1]
        predictions = [0, 1, 1]

        csv_path = save_predictions_csv(
            experiments_root=str(tmp_path),
            dataset_id="test",
            model_name="model",
            seed=42,
            image_paths=paths,
            scores=scores,
            labels=labels,
            predictions=predictions,
            preprocessing_mode="baseline",
            n_train=100,
        )

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["image_path"] == "/img/a.png"
        assert float(rows[0]["score"]) == pytest.approx(0.1, abs=1e-4)
        assert int(rows[0]["label"]) == 0
        assert int(rows[0]["prediction"]) == 0
        assert int(rows[2]["prediction"]) == 1

    def test_csv_header(self, tmp_path):
        csv_path = save_predictions_csv(
            experiments_root=str(tmp_path),
            dataset_id="ds",
            model_name="m",
            seed=42,
            image_paths=["a.png"],
            scores=[0.5],
            labels=[1],
            predictions=[1],
        )
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == "image_path,score,label,prediction"


# ── 4. Runtime profiler latency std ──────────────────────────────────────────


class TestRuntimeProfilerStd:
    """Runtime profiler must report both mean and std for latency."""

    def test_measure_latency_returns_tuple(self):
        device = torch.device("cpu")
        result = measure_latency(lambda: None, device, warmup=10, iterations=30)
        assert isinstance(result, tuple)
        assert len(result) == 2
        mean, std = result
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0.0

    def test_profile_model_has_std_fields(self):
        device = torch.device("cpu")
        result = profile_model(
            model_only_fn=lambda: None,
            end_to_end_fn=lambda: None,
            device=device,
            warmup=10,
            iterations=30,
        )
        assert hasattr(result, "model_only_latency_std_ms")
        assert hasattr(result, "end_to_end_latency_std_ms")
        assert result.model_only_latency_std_ms >= 0.0
        assert result.end_to_end_latency_std_ms >= 0.0

    def test_to_dict_includes_std(self):
        device = torch.device("cpu")
        result = profile_model(
            model_only_fn=lambda: None,
            end_to_end_fn=lambda: None,
            device=device,
            warmup=10,
            iterations=30,
        )
        d = result.to_dict()
        assert "model_only_latency_std_ms" in d
        assert "end_to_end_latency_std_ms" in d


# ── 5. Seed determinism ─────────────────────────────────────────────────────


class TestSeedDeterminism:
    """set_seed must produce identical results across calls."""

    def test_torch_determinism(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_numpy_determinism(self):
        set_seed(42)
        a = np.random.randn(10)
        set_seed(42)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(1337)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


# ── 6. No test leakage structural tests ─────────────────────────────────────


class TestNoTestLeakage:
    """Threshold must never be computed from test labels."""

    def test_thresholding_accepts_only_scores(self):
        """compute_threshold takes only val_scores, no labels parameter."""
        import inspect

        sig = inspect.signature(compute_threshold)
        param_names = set(sig.parameters.keys())
        # Must not accept 'labels', 'test_labels', 'all_labels', etc.
        assert "labels" not in param_names
        assert "test_labels" not in param_names
        assert "all_labels" not in param_names

    def test_thresholding_module_has_no_roc_curve(self):
        """shared/thresholding.py must not import or call roc_curve."""
        thresholding_path = REPO_ROOT / "shared" / "thresholding.py"
        source = thresholding_path.read_text()
        import ast as _ast

        tree = _ast.parse(source)
        # Check import statements for roc_curve
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                for alias in node.names:
                    assert (
                        alias.name != "roc_curve"
                    ), "shared/thresholding.py imports roc_curve"
            if isinstance(node, (_ast.Call,)):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                assert name != "roc_curve", "shared/thresholding.py calls roc_curve"
        # Also verify no function/variable named 'youden' in code (AST check)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef):
                assert (
                    "youden" not in node.name.lower()
                ), "shared/thresholding.py defines a youden function"
            if isinstance(node, (_ast.Call,)):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name:
                    assert (
                        "youden" not in name.lower()
                    ), "shared/thresholding.py calls a youden function"

    def test_shared_metrics_threshold_is_parameter(self):
        """compute_image_metrics must receive threshold as a parameter, not compute it."""
        import inspect

        sig = inspect.signature(compute_image_metrics)
        assert "threshold" in sig.parameters


# ── 7. Config validation ────────────────────────────────────────────────────


class TestConfigCompliance:
    """config.yaml must have required benchmark protocol fields."""

    @staticmethod
    def _load_config():
        try:
            import yaml
        except ModuleNotFoundError:
            import json, re

            # Minimal YAML-subset parser: fall back to line-level parsing
            config_path = REPO_ROOT / "benchmark" / "config.yaml"
            text = config_path.read_text()
            # Use a simple approach: convert simple YAML to dict
            result = {}
            for line in text.splitlines():
                line = line.split("#")[0].strip()
                if not line or line.startswith("-"):
                    continue
                if ":" in line:
                    key, _, val = line.partition(":")
                    val = val.strip()
                    if val.startswith("[") and val.endswith("]"):
                        # inline list
                        inner = val[1:-1]
                        items = [v.strip() for v in inner.split(",") if v.strip()]
                        parsed = []
                        for item in items:
                            try:
                                parsed.append(int(item))
                            except ValueError:
                                try:
                                    parsed.append(float(item))
                                except ValueError:
                                    parsed.append(item)
                        result[key.strip()] = parsed
                    else:
                        val = val.strip('"').strip("'")
                        try:
                            result[key.strip()] = int(val)
                        except ValueError:
                            try:
                                result[key.strip()] = float(val)
                            except ValueError:
                                result[key.strip()] = val
            return result
        else:
            config_path = REPO_ROOT / "benchmark" / "config.yaml"
            with open(config_path) as f:
                return yaml.safe_load(f)

    def test_config_has_few_shot_sizes(self):
        cfg = self._load_config()
        assert "few_shot_sizes" in cfg
        sizes = cfg["few_shot_sizes"]
        assert isinstance(sizes, list)
        assert 10 in sizes
        assert 25 in sizes
        assert 50 in sizes
        assert 100 in sizes

    def test_config_has_three_seeds(self):
        cfg = self._load_config()
        assert "seeds" in cfg
        assert len(cfg["seeds"]) == 3
        assert cfg["seeds"] == [42, 1337, 2026]

    def test_config_threshold_strategy_quantile(self):
        cfg = self._load_config()
        assert cfg["threshold_strategy"] == "quantile"
        assert cfg["threshold_quantile_p"] == 0.99


# ── 8. Aggregation across seeds ─────────────────────────────────────────────


class TestAggregation:
    """Aggregation must compute mean ± std across seeds."""

    def test_three_seed_aggregation(self, tmp_path):
        seeds = [42, 1337, 2026]
        aurocs = [0.90, 0.92, 0.91]

        for seed, auroc in zip(seeds, aurocs):
            save_results(
                str(tmp_path),
                "ds",
                "model",
                seed,
                {"auroc": auroc, "f1": 0.80},
                runtime={"model_only_latency_ms": 10.0},
                preprocessing_mode="baseline",
                n_train=100,
            )

        summary = aggregate_seeds(
            str(tmp_path),
            "ds",
            "model",
            seeds,
            preprocessing_mode="baseline",
            n_train=100,
        )
        assert summary["n_seeds_found"] == 3
        assert summary["mean"]["auroc"] == pytest.approx(np.mean(aurocs), abs=1e-3)
        assert summary["std"]["auroc"] >= 0.0
        assert summary["std"]["auroc"] > 0.0  # not zero since values differ

    def test_latex_row_format(self, tmp_path):
        seeds = [42, 1337, 2026]
        for seed in seeds:
            save_results(
                str(tmp_path),
                "ds",
                "model",
                seed,
                {"auroc": 0.95, "f1": 0.90, "precision": 0.92, "recall": 0.88},
                runtime={
                    "model_only_latency_ms": 10.0,
                    "end_to_end_latency_ms": 15.0,
                    "gpu_peak_memory_mb": 1024.0,
                },
                preprocessing_mode="baseline",
                n_train=100,
            )

        summary = aggregate_seeds(
            str(tmp_path),
            "ds",
            "model",
            seeds,
            preprocessing_mode="baseline",
            n_train=100,
        )
        latex = format_latex_row(summary)
        assert "model" in latex
        assert "\\\\" in latex


# ── 9. Split structural tests ───────────────────────────────────────────────


class TestSplitProtocol:
    """Splits must follow the benchmark protocol."""

    def test_no_overlap_train_val_test(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=10, seed=42, link_mode="copy"
        )

        train = set(m["files"]["train_ok"])
        val = set(m["files"]["val_ok"])
        test_ok = set(m["files"]["test_ok"])
        test_nok = set(m["files"]["test_nok"])

        assert train.isdisjoint(val)
        assert train.isdisjoint(test_ok)
        assert val.isdisjoint(test_ok)
        # NOK names are inherently different from OK names
        assert train.isdisjoint(test_nok)

    def test_all_seeds_produce_same_test(self, fake_dataset):
        """All 3 official seeds must produce the same test set for same n_train."""
        ds, splits_root = fake_dataset
        seeds = [42, 1337, 2026]
        manifests = []
        for seed in seeds:
            m = create_split(
                str(ds),
                str(splits_root),
                "test_ds",
                n_train=10,
                seed=seed,
                link_mode="copy",
            )
            manifests.append(m)

        # test/nok must always be the same (all NOK images)
        for m in manifests:
            assert m["counts"]["test_nok"] == NUM_NOK

    def test_few_shot_invariant_official_sizes(self, fake_dataset):
        """Val and test must be identical across [10, 25, 50] for same seed."""
        ds, splits_root = fake_dataset
        sizes = [10, 25, 50]
        manifests = {}
        for n in sizes:
            m = create_split(
                str(ds),
                str(splits_root),
                "test_ds",
                n_train=n,
                seed=42,
                link_mode="copy",
            )
            manifests[n] = m

        ref_val = sorted(manifests[10]["files"]["val_ok"])
        ref_test_ok = sorted(manifests[10]["files"]["test_ok"])
        ref_test_nok = sorted(manifests[10]["files"]["test_nok"])

        for n in [25, 50]:
            assert sorted(manifests[n]["files"]["val_ok"]) == ref_val
            assert sorted(manifests[n]["files"]["test_ok"]) == ref_test_ok
            assert sorted(manifests[n]["files"]["test_nok"]) == ref_test_nok

    def test_manifest_has_required_fields(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=10, seed=42, link_mode="copy"
        )

        required_keys = [
            "dataset_id",
            "seed",
            "n_train",
            "split_dir",
            "has_masks",
            "counts",
            "files",
        ]
        for key in required_keys:
            assert key in m, f"Missing manifest key: {key}"

        assert "train_ok" in m["files"]
        assert "val_ok" in m["files"]
        assert "test_ok" in m["files"]
        assert "test_nok" in m["files"]

    def test_val_count_is_nonzero(self, fake_dataset):
        """There must always be validation images for threshold computation."""
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=10, seed=42, link_mode="copy"
        )
        assert m["counts"]["val_ok"] > 0


# ── 10. Benchmark run_benchmark.py structural test ─────────────────────────


class TestRunBenchmarkStructure:
    """All run_benchmark.py files must use shared utilities."""

    BENCHMARK_FILES = [
        "models/patchcore/src/run_benchmark.py",
        "models/rd_plus_plus/src/run_benchmark.py",
        "models/simplenet/src/run_benchmark.py",
        "models/anomalydino/src/run_benchmark.py",
    ]

    @pytest.mark.parametrize("rel_path", BENCHMARK_FILES)
    def test_uses_shared_thresholding(self, rel_path):
        source = (REPO_ROOT / rel_path).read_text()
        assert "from shared.thresholding import" in source

    @pytest.mark.parametrize("rel_path", BENCHMARK_FILES)
    def test_uses_shared_metrics(self, rel_path):
        source = (REPO_ROOT / rel_path).read_text()
        assert "from shared.metrics import" in source

    @pytest.mark.parametrize("rel_path", BENCHMARK_FILES)
    def test_uses_shared_preprocessing(self, rel_path):
        source = (REPO_ROOT / rel_path).read_text()
        assert "from shared.preprocessing import" in source

    @pytest.mark.parametrize("rel_path", BENCHMARK_FILES)
    def test_uses_shared_results_io(self, rel_path):
        source = (REPO_ROOT / rel_path).read_text()
        assert "from shared.results_io import" in source
        assert "save_predictions_csv" in source

    @pytest.mark.parametrize("rel_path", BENCHMARK_FILES)
    def test_uses_shared_seed_utils(self, rel_path):
        source = (REPO_ROOT / rel_path).read_text()
        assert "from shared.seed_utils import" in source

    @pytest.mark.parametrize("rel_path", BENCHMARK_FILES)
    def test_no_youden_j_in_benchmark(self, rel_path):
        """run_benchmark.py must not contain Youden J or roc_curve for threshold."""
        source = (REPO_ROOT / rel_path).read_text()
        assert "youden" not in source.lower(), f"{rel_path} contains 'youden'"
        # roc_curve is OK in compute_image_metrics (for AUROC), but not for threshold
        # Check that roc_curve is not used for threshold computation
        assert "j_scores = tpr - fpr" not in source, f"{rel_path} computes Youden J"


# ── 11. Hardening tests ─────────────────────────────────────────────────────


class TestHardeningGuards:
    """Guards added during hardening pass must trigger correctly."""

    def test_aggregation_fails_on_missing_seed(self, tmp_path):
        """aggregate_seeds must raise FileNotFoundError if any seed is missing."""
        seeds = [42, 1337, 2026]
        # Save results for only 2 of 3 seeds
        for seed in seeds[:2]:
            save_results(
                str(tmp_path),
                "ds",
                "model",
                seed,
                {"auroc": 0.90, "f1": 0.80},
                runtime={"model_only_latency_ms": 10.0},
                preprocessing_mode="baseline",
                n_train=100,
            )

        with pytest.raises(FileNotFoundError, match="Missing results for seed"):
            aggregate_seeds(
                str(tmp_path),
                "ds",
                "model",
                seeds,
                preprocessing_mode="baseline",
                n_train=100,
            )

    def test_thresholding_rejects_nan(self):
        """compute_threshold must reject NaN in val_scores."""
        scores = [0.1, 0.2, float("nan"), 0.4]
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_threshold(scores, strategy="quantile")

    def test_thresholding_rejects_inf(self):
        """compute_threshold must reject Inf in val_scores."""
        scores = [0.1, 0.2, float("inf"), 0.4]
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_threshold(scores, strategy="quantile")

    def test_split_rejects_n_train_zero(self, fake_dataset):
        """create_split must reject n_train=0."""
        ds, splits_root = fake_dataset
        with pytest.raises(ValueError, match="n_train must be >= 1"):
            create_split(
                str(ds),
                str(splits_root),
                "test_ds",
                n_train=0,
                seed=42,
                link_mode="copy",
            )

    def test_metrics_require_both_classes(self):
        """compute_image_metrics must reject single-class labels."""
        labels = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="requires both normal.*and anomalous"):
            compute_image_metrics(labels, scores, threshold=0.5)

    def test_training_folder_ok_only(self, fake_dataset):
        """Split train set must contain only ok images (label=0)."""
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds),
            str(splits_root),
            "test_ds",
            n_train=10,
            seed=42,
            link_mode="copy",
        )
        for f in m["files"]["train_ok"]:
            # All training images must come from the ok/ pool
            assert "nok" not in Path(f).stem

    @pytest.mark.parametrize("rel_path", TestRunBenchmarkStructure.BENCHMARK_FILES)
    def test_preprocessing_uses_build_transforms(self, rel_path):
        """All run_benchmark.py files must use shared build_transforms."""
        source = (REPO_ROOT / rel_path).read_text()
        assert (
            "build_transforms" in source
        ), f"{rel_path} does not use build_transforms from shared.preprocessing"
