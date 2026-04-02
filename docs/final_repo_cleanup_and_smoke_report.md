# Final Repository Cleanup and Smoke Test Report

**Date:** 2026-03-30
**Branch:** velke-zmeny
**Dataset:** bottle (seed=42, n_train=159, mode=main, preprocessing=baseline)

---

## 1. Cleanup Actions Performed

### Deleted — stale experiment artifacts

| Path | Reason |
|---|---|
| `experiments/bottle/simplenet/{42,1337,2026}/` | Old non-canonical layout (checkpoint-only dirs: `model.pth`, `model_best.pth`). Canonical results exist under `mode=baseline/` layout. |
| `experiments/wood/simplenet/{42,1337,2026}/` | Same — old non-canonical layout. |
| `experiments/pcb1/simplenet/{42,1337}/` | Same — old non-canonical layout. |
| `experiments/pcb1/rd_plus_plus/` | Empty directory tree (no files). |

### Deleted — transient caches

| Path | Reason |
|---|---|
| `shared/__pycache__/` | Python bytecode cache. |
| `benchmark/__pycache__/` | Python bytecode cache. |
| `tests/__pycache__/` | Python bytecode cache. |
| `models/simplenet/src/__pycache__/` | Python bytecode cache. |
| `models/anomalydino/src/__pycache__/` | Python bytecode cache. |
| `.pytest_cache/` | Pytest transient cache. |

### Deleted — intermediate development docs

| Path | Reason |
|---|---|
| `docs/anomalydino_gap_analysis.md` | Intermediate development artifact (gap analysis stage); superseded by final report. |
| `docs/anomalydino_repo_bootstrap.md` | Intermediate development artifact (bootstrap stage); superseded by final report. |

### Approximate space freed

- ~60 MB from old SimpleNet checkpoint dirs
- ~0 from empty pcb1/rd++ dirs
- Negligible from caches and docs

---

## 2. Deliberately Preserved

- All canonical experiment results under `experiments/{dataset}/{model}/mode=baseline/` layout (all 3 datasets × all seeds × all available models)
- All split manifests under `splits/`
- All dataset files under `datasets/`
- All source code under `models/`, `shared/`, `benchmark/`, `tests/`
- Canonical docs: `benchmark_protocol.md`, `patchcore_verification_report.md`, `anomalydino_final_report.md`, `anomalydino_final_control.md`, `anomalydino_repair_note.md`
- Config files: `pyproject.toml`, `requirements.txt`, `setup.sh`, `README.md`, `.gitignore`, `benchmark/config.yaml`
- RD++ model checkpoints (`model_best.pth`) — required by the benchmark pipeline

---

## 3. Smoke Test Executed

**Command (first 3 models):**
```
PYTHONPATH=. python benchmark/run.py --dataset bottle --mode main --all-models --seeds 42 -v
```

**Command (AnomalyDINO retry after blocker fix):**
```
PYTHONPATH=. python benchmark/run.py --dataset bottle --mode main --model anomalydino --seeds 42 -v
```

**Parameters:** dataset=bottle, n_train=159 (from registry), seed=42, preprocessing_mode=baseline, threshold_strategy=quantile (p=0.98)

---

## 4. Minimal Fix Applied (Blocker)

**Issue:** AnomalyDINO failed on first run with `PermissionError: [Errno 13] Permission denied: '/tmp/.cache/torch/hub/trusted_list'`.

**Root cause:** The Docker named volume `anomalydino_dino_hub_cache` was created with root ownership, but the container runs as UID 1000:1000. The initial `torch.hub.load()` call cannot write the trust file.

**Fix:** One-time permission correction on the Docker volume:
```
docker run --rm -v anomalydino_dino_hub_cache:/data alpine chown -R 1000:1000 /data
```

**Scope:** Infrastructure-only (Docker volume permissions). No source code changes. No functionality changes.

---

## 5. Model Status

| Model | AUROC | AP | Precision | Recall | F1 | Latency (ms) | GPU Peak (MB) | Fit Time (s) | Status |
|---|---|---|---|---|---|---|---|---|---|
| PatchCore | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5.66 model / 20.07 e2e | 982.4 | 68.2 | **PASS** |
| RD++ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 7.74 model / 23.68 e2e | 1020.2 | 1676.1 | **PASS** |
| SimpleNet | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.06 model / 19.22 e2e | 426.8 | 274.0 | **PASS** |
| AnomalyDINO | 1.0000 | 1.0000 | 0.9692 | 1.0000 | 0.9844 | 5.32 model / 20.14 e2e | 614.9 | 12.5 | **PASS** |

### Artifact Validation

For each model, verified:
- `results.json` exists, is valid JSON, contains dataset_id=bottle, seed=42, n_train=159
- `predictions.csv` exists with 84 rows (1 header + 83 test images: 20 OK + 63 NOK)
- `summary.json` exists (aggregation over single seed)
- Metrics fields populated (no NaN)
- Runtime fields populated (latency, GPU memory, fit time)
- Config block present with correct parameters
- Canonical output path: `experiments/bottle/{model}/mode=baseline/n_train=159/seed=42/`

---

## 6. Final Conclusion

**READY FOR CLEAN BENCHMARK START**

The repository is clean, all four model pipelines execute successfully end-to-end through the official benchmark orchestrator, and all output artifacts are structurally correct and populated.
