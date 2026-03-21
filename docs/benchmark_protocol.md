# Benchmark Protocol

This document specifies the evaluation protocol used by the benchmark.
It ensures fair, reproducible comparison across all anomaly detection models.

---

## 1. Split methodology

The split manager (`shared/split_manager.py`) creates deterministic
train / val / test partitions from a flat dataset layout:

```
datasets/<dataset_id>/
├── ok/       ← all normal images
├── nok/      ← all defective images
└── masks/    ← optional pixel-level masks
```

### Pool allocation

Given `seed` and `n_train`:

1. Shuffle all OK images with `random.Random(seed)`
2. Allocate pools in order:
   - **test/ok** — first `min(200, total_ok)` images
   - **val/ok** — next `max(1, 15% of total_ok)` images
   - **train/ok** — next `n_train` images from the remainder
3. **test/nok** — all defective images (always)

### Few-shot invariant

For a fixed `(dataset_id, seed)`, the val and test pools are determined
**before** `n_train` is applied. Changing `n_train` only changes
which images go into `train/ok`; val and test remain constant.

This is critical: it means AUROC at `n_train=10` and `n_train=100`
are evaluated on the exact same test set.

### Manifest

Each split produces a JSON manifest at:

```
splits/<dataset_id>/seed<N>/n<train>/manifest.json
```

The manifest records file lists, counts, seed, and link mode.
All models consume the same manifest — no model-local split logic.

---

## 2. Preprocessing

Unified transform builder: `shared/preprocessing.py`.

### Baseline mode (official benchmark default)

```
Resize(shortest_edge=256, bilinear)
→ CenterCrop(224)
→ ToTensor
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Why 224 is the default

- Matches ImageNet pretraining resolution for most backbones
- Standard for WideResNet-50-2, ResNet-18 architectures
- Gives all models equal footing regardless of native preference
- 448 available via `high_accuracy` mode for DINOv2-native resolution

### Alternative modes

| Mode | Pipeline | Final size |
|---|---|---|
| `high_accuracy` | Resize(448×448) → ToTensor → Normalize | 448×448 |
| `edge_safe` | ResizeKeepAspect + PadToSquare(224) → ToTensor → Normalize | 224×224 |

### No augmentations

Transforms are fully deterministic — no random flips, rotations, or
color jitter. This ensures reproducibility and fair comparison.

---

## 3. Thresholding

Module: `shared/thresholding.py`

### Rule: val-only

The anomaly threshold is computed **exclusively** from val/ok scores
(images known to be normal). Test labels are never used for threshold selection.

This prevents information leakage from test data into the decision boundary.

### Default strategy: quantile

```python
threshold = np.quantile(val_ok_scores, p=0.99)
```

Interpretation: a score is classified as anomalous if it exceeds the 99th
percentile of normal validation scores.

### Alternative strategies

| Strategy | Formula | When to use |
|---|---|---|
| `quantile` (default) | `np.quantile(scores, p)` | General purpose |
| `max` | `max(val_ok_scores)` | Conservative (lowest false-positive rate) |
| `k_sigma` | `mean + k × std` | Gaussian-ish score distributions |

### What is NOT done

- No `roc_curve` on test data
- No Youden J statistic on test data
- No optimal threshold search on test labels

These would inflate metrics and are methodologically incorrect for
unsupervised anomaly detection evaluation.

---

## 4. Metrics

Module: `shared/metrics.py`

### Image-level

| Metric | Description |
|---|---|
| **AUROC** | Area under ROC curve (threshold-independent ranking) |
| **Precision** | TP / (TP + FP) at the val-derived threshold |
| **Recall** | TP / (TP + FN) at the val-derived threshold |
| **F1** | Harmonic mean of precision and recall |

### Pixel-level

| Metric | Description |
|---|---|
| **AU-PRO** | Per-Region Overlap integrated up to FPR=0.3 |

AU-PRO is reported **only** when:
1. The model produces pixel-level anomaly maps
2. The dataset provides pixel-level ground-truth masks

Otherwise `"N/A"` with a reason string.

Currently: only RD++ produces anomaly maps. PatchCore, SimpleNet, and
AnomalyDINO operate at image level only.

---

## 5. Runtime profiling

Module: `shared/runtime_profiler.py`

### Protocol

- Batch size: **1** (single-image inference)
- Warm-up: ≥10 forward passes (discarded)
- Measurement: ≥30 forward passes
- `torch.cuda.synchronize()` before and after each pass
- GPU peak memory measured via `torch.cuda.max_memory_allocated()`

### Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `model_latency_mean` | ms | Mean model-only inference time |
| `model_latency_std` | ms | Std dev of model-only inference time |
| `e2e_latency_mean` | ms | Mean end-to-end (load + preprocess + infer) |
| `e2e_latency_std` | ms | Std dev of end-to-end time |
| `gpu_peak_memory_mb` | MB | Peak GPU memory during inference |

`measure_latency()` returns `(mean, std)` — both values are stored
in the `RuntimeResult` dataclass and persisted in the JSON results.

---

## 6. Results & auditability

Module: `shared/results_io.py`

### Per-seed JSON

`save_results()` persists a JSON file per `(model, dataset, seed, n_train)`.
Each file includes an `"environment"` block with:

| Field | Example |
|---|---|
| `git_sha` | `4efbe247…` |
| `timestamp` | ISO-8601 UTC |
| `platform` | `Linux-6.x-x86_64` |
| `python_version` | `3.10.12` |
| `torch_version` | `2.1.0+cu121` |
| `cuda_available` | `true` |
| `gpu_name` | `NVIDIA RTX 4090` |
| `gpu_memory_total_mb` | `24564` |

This makes every result traceable to a specific commit and hardware state.

### Per-image predictions CSV

`save_predictions_csv()` writes a CSV alongside the JSON results:

```
image_path,score,label,prediction
datasets/casting/ok/0001.png,0.12,0,0
datasets/casting/nok/0042.png,0.87,1,1
```

This enables post-hoc analysis, error inspection, and threshold sweep
without re-running inference.

---

## 7. Multi-seed aggregation

Module: `shared/aggregate.py`

Every experiment runs across **3 seeds** (default: 42, 1337, 2026).

Final reported values are **mean ± std** across seeds for all metrics.
This captures variance from:
- Random split (different train images per seed)
- Model initialization (for trained models)
- Training stochasticity

### LaTeX output

`format_latex_row()` produces a table row:

```
model & AUROC & F1 & Latency & GPU Memory \\
```

---

## 8. Few-shot evaluation

Configuration key: `few_shot_sizes` in `benchmark/config.yaml`.

Default sizes: **[10, 25, 50, 100]**.

When invoked with the `--few-shot` flag:

```bash
python benchmark/run.py --dataset casting --model patchcore --few-shot
```

The orchestrator runs the full `(model × seed × n_train)` grid for all
configured few-shot sizes. This produces comparable curves showing how
model performance scales with the number of training samples.

The few-shot invariant (Section 1) guarantees that val and test sets are
identical across all `n_train` values for a given `(dataset_id, seed)`.

---

## 9. Results storage

Module: `shared/results_io.py`

Per-seed results are stored as JSON at:

```
experiments/<dataset_id>/<model_name>/<seed>/results.json
```

Each file contains:
- Image-level metrics (AUROC, F1, precision, recall)
- Pixel-level metrics (AU-PRO or "N/A")
- Runtime profiling results
- Full configuration snapshot (seed, n_train, mode, threshold)
- Timestamp

---

## 10. Troubleshooting

### "Not enough OK images"

The split manager requires at least `test_ok(200) + val_ok(15%) + n_train`
OK images. With small datasets, reduce `n_train` or use a larger dataset.

### Different results across runs

Ensure `shared/seed_utils.set_seed()` is called before any model operation.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for full CUDA determinism.

### Docker GPU issues

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Model produces anomaly maps but AU-PRO is N/A

Check that `datasets/<id>/masks/` exists with filenames matching `nok/` stems.
