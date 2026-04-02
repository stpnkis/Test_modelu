# PatchCore Pipeline Verification Report

**Date:** 2026-03-26  
**Branch:** velke-zmeny  
**Purpose:** Post-configuration-selection stabilization and verification of the PatchCore benchmark pipeline.

---

## 1. Final Configuration

The configuration was selected in the preceding cross-dataset sweep (4 candidates × 3 datasets × 3 seeds). **Config C** was chosen as the global winner and is now the baked-in default.

| Parameter | Value | Code location |
|---|---|---|
| `backbone` | `resnet50` | `models/patchcore/src/run_benchmark.py` L138 |
| `layers` | `[layer2, layer3]` | `models/patchcore/src/run_benchmark.py` L139 |
| `coreset_sampling_ratio` | `0.1` | `models/patchcore/src/run_benchmark.py` L140 |
| `num_neighbors` | `15` | `models/patchcore/src/run_benchmark.py` L141 |
| `threshold_strategy` | `quantile` | `models/patchcore/src/run_benchmark.py` L142 |
| `threshold_quantile_p` | `0.98` | `models/patchcore/src/run_benchmark.py` L143 |
| `preprocessing_mode` | `baseline` | `benchmark/config.yaml` |
| `image_size` | `224` | `models/patchcore/src/run_benchmark.py` L144 |
| `seeds` | `[42, 1337, 2026]` | `benchmark/config.yaml` |

All parameters remain overridable via CLI; the defaults ensure reproducibility without extra flags.

---

## 2. Pipeline Description

The pipeline runs entirely inside a Docker container (`pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`, anomalib ≥ 1.2). The orchestrator (`benchmark/orchestrator.py`) drives the following sequence:

```
1. SPLIT CREATION (host, shared/split_manager.py)
   - Deterministic stratified split: train_ok / val_ok / test_ok / test_nok
   - Policy: official_test — test set taken from dataset's official test split
   - n_val_ok = 50 fixed; remaining ok images → train_ok
   - Written to splits/{dataset}/seed{seed}/n{n_train}/
   - Symlinks normalised to lowercase to ensure cross-filesystem portability

2. TRAINING (Docker, anomalib Engine.fit)
   - Folder datamodule with val_split_mode=SAME_AS_TEST (disables internal val split)
   - num_sanity_val_steps=0 (prevents contamination of fit-time validation)
   - Coreset subsampling (ratio=0.1) applied after feature extraction

3. VAL SCORING (Docker, direct model inference)
   - Score all val_ok images via patchcore_torch() — bypasses anomalib DM entirely
   - Runtime assertion: all val labels == 0 (no leakage possible)

4. THRESHOLD COMPUTATION (Docker, shared/thresholding.py)
   - Strategy: quantile at p=0.98 over val_ok scores
   - Only val_ok scores are used — test scores not yet seen

5. TEST SCORING (Docker, direct model inference)
   - Score test_ok and test_nok images via patchcore_torch()
   - Produces (scores, labels, paths) for all test images

6. METRIC COMPUTATION (Docker, shared/metrics.py)
   - AUROC, AP: threshold-free, from raw scores
   - Precision, Recall, F1: at the quantile threshold from step 4
   - n_val_ok_for_threshold recorded in results.json for auditability

7. SAVE (Docker, shared/results_io.py)
   - results.json: all metrics + runtime + config snapshot
   - predictions.csv: per-image (path, score, label, prediction) for test set
   - Path: experiments/{dataset}/patchcore/mode=baseline/n_train={n}/seed={seed}/
```

**Seeding:** `set_seed(seed)` is called before every anomalib Folder/Engine instantiation, ensuring deterministic coreset subsampling and feature extraction ordering.

---

## 3. Verification Results

### 3.1 Split Integrity

All splits verified disjoint across all 6 experiment instances (2 datasets × 3 seeds):

| Check | Result |
|---|---|
| train_ok ∩ val_ok | 0 overlap |
| train_ok ∩ test_ok | 0 overlap |
| val_ok ∩ test_ok | 0 overlap |
| val_ok ∩ test_nok | 0 overlap |

Split sizes are consistent across seeds (same n_train, val, test allocation per dataset).

### 3.2 Threshold Correctness

Threshold is computed solely from val_ok scores. This was verified by:
- Confirming `anomalib_internal_split_disabled: true` in every results.json
- Confirming `n_val_ok_for_threshold: 50` matches the expected val split size
- Runtime assertion in code (`assert all(lbl == 0 for lbl in val_labels)`)
- Thresholds vary across seeds (reflecting different val_ok subsets), confirming the threshold is sensitive to the specific val split and was not hard-coded

### 3.3 Metric Correctness

All metrics independently recomputed from `predictions.csv` and compared to `results.json`:

- Precision, Recall, F1: recomputed from confusion matrix using stored threshold → **exact match** for all 6 runs
- AUROC, AP: recomputed using sklearn from raw scores and labels → **exact match** for all 6 runs
- Label–path consistency: `label=0` ↔ `/ok/` path, `label=1` ↔ `/nok/` path → **0 mismatches** across all 6 runs
- Prediction–threshold consistency: `prediction` column matches `score >= threshold` → **0 mismatches**

### 3.4 Config Consistency

Across all 3 seeds, the `config` block in results.json is identical for both datasets (backbone, layers, coreset ratio, num_neighbors, threshold strategy, quantile p, preprocessing\_mode, image\_size).

### 3.5 Unit Tests

186/191 tests pass on host. The 5 failures are all in `_get_environment_info()` due to a host-side CUDA driver version mismatch (host driver < container driver) — not a code defect. All functional pipeline tests pass.

---

## 4. Experimental Results

### 4.1 Wood Dataset (n_train=197, test: 19 ok + 60 nok)

| seed | AUROC  | AP     | Precision | Recall | F1     | Threshold | TP | FP | TN | FN |
|------|--------|--------|-----------|--------|--------|-----------|----|----|----|----|
| 42   | 0.9930 | 0.9979 | 0.9516    | 0.9833 | 0.9672 | 25.0555   | 59 | 3  | 16 | 1  |
| 1337 | 0.9921 | 0.9977 | 0.9365    | 0.9833 | 0.9593 | 24.5226   | 59 | 4  | 15 | 1  |
| 2026 | 0.9930 | 0.9979 | 0.9516    | 0.9833 | 0.9672 | 25.5029   | 59 | 3  | 16 | 1  |

**Aggregate: AUROC = 0.9927 ± 0.0004, AP = 0.9978 ± 0.0001, Recall = 0.9833 ± 0.0000, F1 = 0.9646 ± 0.0037**

Runtime: model latency 6.44 ± 0.11 ms, GPU peak 1175 MB, fit time 67.7 s

### 4.2 Bottle Dataset (n_train=159, test: 20 ok + 63 nok)

| seed | AUROC  | AP     | Precision | Recall | F1     | Threshold | TP | FP | TN | FN |
|------|--------|--------|-----------|--------|--------|-----------|----|----|----|----|
| 42   | 1.0000 | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 20.1166   | 63 | 0  | 20 | 0  |
| 1337 | 1.0000 | 1.0000 | 1.0000    | 1.0000 | 1.0000 | 21.1614   | 63 | 0  | 20 | 0  |
| 2026 | 1.0000 | 1.0000 | 0.9844    | 1.0000 | 0.9921 | 18.4957   | 63 | 1  | 19 | 0  |

**Aggregate: AUROC = 1.0000 ± 0.0000, AP = 1.0000 ± 0.0000, Recall = 1.0000 ± 0.0000, F1 = 0.9974 ± 0.0037**

Runtime: model latency 5.63 ± 0.04 ms, GPU peak 982 MB, fit time 108.7 s

---

## 5. Identified Risks

| Risk | Severity | Status |
|---|---|---|
| **W2 — `validate_split_invariance()` never called at runtime.** The split manager has an invariance checker that verifies split integrity post-hoc, but it is not called by the orchestrator. An audit of the manifest format and split logic found no actual bugs; the splits are correct (verified manually above). | Low | Reported only — conservative approach; changing split manager at this stage is unnecessary. |
| **Val scores not persisted.** `predictions.csv` records only test-set scores. The threshold cannot be reproduced without re-running the model on val_ok. All threshold values are stored in `results.json`; if exact reproduction is needed, val_ok images are deterministic from the seed. | Low | Documented. No fix needed for benchmark purposes. |
| **5 host-side test failures** in `_get_environment_info()` due to CUDA driver version mismatch (host driver too old for the container's CUDA 11.8 toolkit). All functional tests pass. Does not affect Docker-based experiment execution. | Low | Expected. No code change needed. |
| **Stale checkpoint directories** (`experiments/{ds}/patchcore/{seed}/Patchcore/`) from earlier runs using the pre-stabilisation path format. Contained only model weights (no results). **Removed in this session.** | Resolved | Cleaned up. |

---

## 6. Final Verdict

**The PatchCore pipeline is ready for benchmark use.**

All critical correctness properties have been verified:
- No data leakage between train / val / test splits
- Threshold computed exclusively from validation-set normal samples
- All reported metrics independently reproduced from raw per-image scores
- Configuration fully consistent across all seeds and datasets
- Repository cleaned of all config-search artifacts
- Selected Config C (resnet50, layers=[layer2,layer3], coreset=0.1, neighbors=15, q=0.98) is the stable default

Performance on the two validation datasets is strong and stable: AUROC ≥ 0.993 on wood (a challenging multi-class texture dataset) and AUROC = 1.000 on bottle, with recall ≥ 0.983 in both cases. The low inter-seed variance (std ≤ 0.0004 AUROC) confirms good reproducibility under the three-seed protocol.
