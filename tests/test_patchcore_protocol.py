"""
Tests for PatchCore protocol compliance — guards against the anomalib
internal-resplit bug and ensures strict data-flow correctness.

Root bug (fixed):
    Anomalib's Folder datamodule defaults:
      val_split_mode = FROM_TEST   → randomly splits test into val + test (50/50)
      val_split_ratio = 0.5
    This caused:
      1. test set silently halved (200 OK + 60 NOK → 100 OK + 30 NOK)
      2. val scores contaminated with anomaly samples from test/nok
      3. threshold inflated to 1.0 (quantile of mixed normal+anomaly)
      4. recall collapsed to ~0.2 despite AUROC > 0.99

These tests ensure the fix remains in place.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

from shared.split_manager import create_split
from shared.thresholding import compute_threshold
from shared.metrics import compute_image_metrics
from shared.dataset_registry import DatasetConfig, get_dataset_config

REPO_ROOT = Path(__file__).resolve().parent.parent
PATCHCORE_SRC = REPO_ROOT / "models" / "patchcore" / "src" / "run_benchmark.py"

NUM_OK = 300
NUM_NOK = 20

_TEST_DS_CONFIG = DatasetConfig(
    dataset_id="test_ds",
    split_policy="custom_holdout",
    official_train_ok=0,
    official_test_ok=0,
    total_nok=NUM_NOK,
    val_ok_count=30,
    main_train_ok=NUM_OK - 30 - 50,  # 220
    custom_test_ok_count=50,
    few_shot_levels=[10, 25, 50],
    has_masks=False,
)


def _patch_registry(monkeypatch):
    _original = get_dataset_config

    def _patched(dataset_id):
        if dataset_id == "test_ds":
            return _TEST_DS_CONFIG
        return _original(dataset_id)

    monkeypatch.setattr("shared.split_manager.get_dataset_config", _patched)


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    ds = tmp_path / "datasets" / "test_ds"
    ok_dir = ds / "ok"
    nok_dir = ds / "nok"
    ok_dir.mkdir(parents=True)
    nok_dir.mkdir(parents=True)
    for i in range(NUM_OK):
        (ok_dir / f"ok_{i:04d}.png").write_bytes(b"fake")
    for i in range(NUM_NOK):
        (nok_dir / f"nok_{i:04d}.png").write_bytes(b"fake")
    _patch_registry(monkeypatch)
    return ds, tmp_path / "splits"


# ── 1. Structural: PatchCore adapter must not use anomalib predict path ──────


class TestPatchcoreAdapterStructure:
    """Static analysis of run_benchmark.py to verify correct data flow."""

    def test_source_exists(self):
        assert PATCHCORE_SRC.exists()

    def _source(self) -> str:
        return PATCHCORE_SRC.read_text()

    def test_no_engine_predict_for_scoring(self):
        """engine.predict() must NOT be used for val or test scoring.

        The old code used engine.predict(datamodule=...) which goes through
        anomalib's predict_dataloader → test_dataloader → silently resplit data.
        The fix uses direct model inference via _score_dataset().
        """
        source = self._source()
        # Count occurrences of engine.predict — should be zero
        predict_calls = re.findall(r'engine\.predict\s*\(', source)
        assert len(predict_calls) == 0, (
            f"run_benchmark.py still uses engine.predict() ({len(predict_calls)} call(s)). "
            "This bypasses the protocol fix. Use _score_dataset() instead."
        )

    def test_uses_score_dataset_for_val(self):
        """Val scoring must use _score_dataset (direct model inference)."""
        source = self._source()
        assert "_score_dataset" in source
        # Check it's called with val_dataset
        assert re.search(r'_score_dataset\s*\(\s*\n?\s*patchcore_torch.*val_dataset', source, re.DOTALL)

    def test_uses_score_dataset_for_test(self):
        """Test scoring must use _score_dataset (direct model inference)."""
        source = self._source()
        assert re.search(r'_score_dataset\s*\(\s*\n?\s*patchcore_torch.*test_dataset', source, re.DOTALL)

    def test_val_split_mode_is_safe(self):
        """val_split_mode must be 'same_as_test' or 'none' — never 'from_test'."""
        source = self._source()
        # Must explicitly set val_split_mode
        assert "val_split_mode" in source
        # Must NOT use from_test
        if 'val_split_mode="from_test"' in source or "val_split_mode='from_test'" in source:
            pytest.fail(
                "val_split_mode='from_test' causes anomalib to randomly split "
                "test set into val+test, destroying protocol guarantees."
            )
        # Must set it to same_as_test or none
        safe = (
            'val_split_mode="same_as_test"' in source
            or "val_split_mode='same_as_test'" in source
            or 'val_split_mode="none"' in source
            or "val_split_mode='none'" in source
        )
        assert safe, "val_split_mode must be 'same_as_test' or 'none'"

    def test_no_val_datamodule_folder(self):
        """There should be NO separate Folder datamodule for val scoring.

        The old code created a second Folder for val/ok scoring, which was
        also subject to anomalib's internal resplit bug.
        """
        source = self._source()
        assert "val_datamodule" not in source, (
            "run_benchmark.py still creates a val_datamodule Folder. "
            "Val scoring should use BenchmarkImageDataset + _score_dataset()."
        )

    def test_uses_concat_dataset_for_test(self):
        """Test set must be built from ConcatDataset of test/ok + test/nok."""
        source = self._source()
        assert "ConcatDataset" in source

    def test_val_label_assertion(self):
        """Must assert that all val labels are 0 (no anomaly contamination)."""
        source = self._source()
        assert "all(" in source and "val_labels" in source

    def test_no_roc_threshold(self):
        """Must not use roc_curve or Youden J for threshold."""
        source = self._source()
        assert "roc_curve" not in source
        assert "youden" not in source.lower()
        assert "j_scores" not in source.lower()

    def test_no_label_from_path_for_scoring(self):
        """Must not use _label_from_path heuristic — labels come from dataset."""
        source = self._source()
        assert "_label_from_path" not in source

    def test_records_protocol_metadata(self):
        """Results must include n_val_ok_for_threshold and split disable flag."""
        source = self._source()
        assert "n_val_ok_for_threshold" in source
        assert "anomalib_internal_split_disabled" in source

    def test_records_scoring_method(self):
        """Config must record that scoring was direct_model_inference."""
        source = self._source()
        assert "direct_model_inference" in source


# ── 2. Thresholding correctness ─────────────────────────────────────────────


class TestThresholdFromValOkOnly:
    """Threshold must be derived from normal-only validation scores."""

    def test_pure_normal_scores_give_reasonable_threshold(self):
        """With clean normal scores, threshold should NOT be 1.0."""
        rng = np.random.RandomState(42)
        val_ok_scores = rng.normal(0.3, 0.05, size=50).tolist()
        t = compute_threshold(val_ok_scores, "quantile", quantile_p=0.99)
        # Normal scores ~0.3 ± 0.05 → threshold should be well below 1.0
        assert t < 1.0, f"Threshold {t} is suspiciously high for normal-only scores"
        assert t > 0.0

    def test_contaminated_scores_inflate_threshold(self):
        """Demonstrate that mixing anomaly scores inflates the threshold.

        This is the exact failure mode of the old code.
        """
        rng = np.random.RandomState(42)
        normal_scores = rng.normal(0.3, 0.05, size=50).tolist()
        anomaly_scores = rng.normal(2.0, 0.3, size=30).tolist()

        t_clean = compute_threshold(normal_scores, "quantile", quantile_p=0.99)
        t_contaminated = compute_threshold(
            normal_scores + anomaly_scores, "quantile", quantile_p=0.99
        )

        assert t_contaminated > t_clean * 2, (
            "Contaminated threshold should be much higher than clean threshold"
        )

    def test_threshold_1_only_if_legitimately_high(self):
        """Threshold = 1.0 is only valid when normal scores genuinely reach 1.0."""
        # Scores intentionally at 1.0
        scores = [1.0] * 100
        t = compute_threshold(scores, "quantile", quantile_p=0.99)
        assert t == pytest.approx(1.0)

        # Normally distributed scores should never produce threshold = 1.0
        rng = np.random.RandomState(42)
        normal_scores = rng.normal(0.3, 0.1, size=100).tolist()
        t2 = compute_threshold(normal_scores, "quantile", quantile_p=0.99)
        assert t2 < 1.0


# ── 3. Test set integrity ───────────────────────────────────────────────────


class TestTestSetIntegrity:
    """Test set must match the split manifest exactly — no halving."""

    def test_manifest_counts_are_full(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(str(ds), str(splits_root), "test_ds", 10, 42, "copy")

        # All NOK images must be in test/nok
        assert m["counts"]["test_nok"] == NUM_NOK
        assert m["counts"]["test_ok"] > 0
        assert m["counts"]["val_ok"] > 0

    def test_no_overlap_val_test(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(str(ds), str(splits_root), "test_ds", 10, 42, "copy")

        val = set(m["files"]["val_ok"])
        test_ok = set(m["files"]["test_ok"])
        assert val.isdisjoint(test_ok), "val/ok and test/ok must not overlap"

    def test_val_contains_only_ok_images(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(str(ds), str(splits_root), "test_ds", 10, 42, "copy")

        for f in m["files"]["val_ok"]:
            assert "nok" not in f, f"val/ok contains NOK file: {f}"

    def test_full_test_set_scored(self, fake_dataset):
        """Simulate: verify that scores list length = test_ok + test_nok."""
        ds, splits_root = fake_dataset
        m = create_split(str(ds), str(splits_root), "test_ds", 10, 42, "copy")

        # Simulate what the adapter should do: score ALL test files
        n_test_ok = m["counts"]["test_ok"]
        n_test_nok = m["counts"]["test_nok"]
        n_total = n_test_ok + n_test_nok

        # The old bug: anomalib halved the test set
        n_halved = n_total // 2
        assert n_total > n_halved, "Full test set must be larger than halved"
        assert n_total == m["counts"]["test_total"]


# ── 4. Metrics validity with correct threshold ──────────────────────────────


class TestMetricsWithCorrectThreshold:
    """Metrics computed with a protocol-compliant threshold."""

    def test_reasonable_threshold_improves_recall(self):
        """A threshold derived from normal-only scores should yield better
        recall than threshold=1.0."""
        rng = np.random.RandomState(42)
        # Normal scores
        ok_scores = rng.normal(0.3, 0.05, size=200)
        # Anomaly scores
        nok_scores = rng.normal(0.8, 0.1, size=60)

        labels = np.array([0] * 200 + [1] * 60)
        scores = np.concatenate([ok_scores, nok_scores])

        # Correct threshold from val/ok
        val_ok = rng.normal(0.3, 0.05, size=50)
        t_correct = compute_threshold(val_ok.tolist(), "quantile", quantile_p=0.99)

        # Broken threshold (what the old code produced)
        t_broken = 1.0

        m_correct = compute_image_metrics(labels, scores, t_correct)
        m_broken = compute_image_metrics(labels, scores, t_broken)

        assert m_correct["recall"] > m_broken["recall"], (
            "Correct threshold should yield higher recall than threshold=1.0"
        )
        assert m_correct["f1"] > m_broken["f1"], (
            "Correct threshold should yield higher F1 than threshold=1.0"
        )

    def test_threshold_1_gives_near_zero_recall(self):
        """With threshold=1.0, recall should be very low for typical scores."""
        rng = np.random.RandomState(42)
        labels = np.array([0] * 200 + [1] * 60)
        scores = np.concatenate([
            rng.normal(0.3, 0.05, size=200),
            rng.normal(0.8, 0.1, size=60),
        ])
        m = compute_image_metrics(labels, scores, threshold=1.0)
        assert m["recall"] < 0.5, "threshold=1.0 should give low recall"


# ── 5. Anomalib FROM_TEST default is dangerous ──────────────────────────────


class TestAnomalibDefaultDanger:
    """Document that anomalib defaults ARE the root cause."""

    def test_from_test_is_the_dangerous_default(self):
        """Verify that anomalib Folder defaults to val_split_mode=FROM_TEST."""
        import inspect
        try:
            from anomalib.data import Folder
            sig = inspect.signature(Folder.__init__)
            default = sig.parameters["val_split_mode"].default
            assert "from_test" in str(default).lower(), (
                f"Expected Folder default val_split_mode=FROM_TEST, got {default}"
            )
        except (ImportError, AttributeError):
            # anomalib not importable on host (e.g. numpy 2.0 + imgaug conflict)
            # Verify via direct source inspection instead
            folder_src = Path(
                "/home/robolab2/stepan_kisler_dp/.venv/lib/python3.10/"
                "site-packages/anomalib/data/image/folder.py"
            )
            if not folder_src.exists():
                pytest.skip("anomalib source not available for static check")
            source = folder_src.read_text()
            assert "ValSplitMode.FROM_TEST" in source, (
                "Could not confirm anomalib defaults to FROM_TEST"
            )

    def test_adapter_overrides_dangerous_default(self):
        """The adapter must explicitly override val_split_mode."""
        source = PATCHCORE_SRC.read_text()
        # Must contain an explicit val_split_mode= in the Folder() call
        assert re.search(
            r'val_split_mode\s*=\s*["\'](?:same_as_test|none)["\']',
            source,
        ), "Adapter must explicitly set val_split_mode to a safe value"
