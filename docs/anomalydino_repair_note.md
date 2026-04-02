# AnomalyDINO — Repair Note (2026-03-30)

## What was wrong

The CLI argument defaults in `models/anomalydino/src/run_benchmark.py`
did not match the frozen final benchmark configuration:

| Parameter | Old default | Frozen config |
|-----------|------------|---------------|
| `--backbone` | `dinov2_vitb14` | `dinov2_vits14` |
| `--layers` | `None` (last layer only) | `8,9,10,11` |
| `--aggregation` | `max` | `topk_mean` |

Additionally, `train.py` had `DEFAULT_BACKBONE = "dinov2_vitb14"`.

The benchmark orchestrator does not pass model-specific CLI args,
so running AnomalyDINO through the official path would have silently
used the wrong configuration.

## What was changed

Two files were edited:

1. **`models/anomalydino/src/run_benchmark.py`**
   - `--backbone` default: `dinov2_vitb14` → `dinov2_vits14`
   - `--layers` default: `None` → `"8,9,10,11"`
   - `--aggregation` default: `max` → `topk_mean`

2. **`models/anomalydino/src/train.py`**
   - `DEFAULT_BACKBONE`: `dinov2_vitb14` → `dinov2_vits14`

No other code was modified.

## What was verified

A benchmark run was executed through the official path with only
standard orchestrator args (no model-specific overrides):

```
PYTHONPATH=. python models/anomalydino/src/run_benchmark.py \
  --dataset-id wood --seed 42 --n-train 197 \
  --preprocessing-mode baseline \
  --threshold-strategy quantile --threshold-quantile-p 0.98 \
  --splits-root splits --experiments-root experiments
```

The resulting `results.json` config block:

```json
{
  "backbone": "dinov2_vits14",
  "layers": "[8, 9, 10, 11]",
  "aggregation": "topk_mean",
  "topk_ratio": 0.01,
  "preprocessing_mode": "baseline",
  "image_size": 224,
  "threshold_strategy": "quantile",
  "threshold_quantile_p": 0.98
}
```

This exactly matches the frozen final configuration. Metrics produced
(AUROC=0.9965, F1=0.9756) are consistent with the original WOOD seed 42
results.

## Verdict

**AnomalyDINO is now READY.** The official benchmark path reproduces
the frozen final configuration automatically without manual overrides.
