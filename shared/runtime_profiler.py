"""
Runtime profiler — consistent latency and GPU memory measurement.

Protocol:
    - batch_size = 1
    - warmup >= 10 iterations
    - measured >= 30 iterations
    - torch.cuda.synchronize() before AND after timing sections
    - GPU peak memory via reset_peak_memory_stats + max_memory_allocated
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class RuntimeResult:
    """Container for runtime profiling results."""

    model_only_latency_ms: float = 0.0
    model_only_latency_std_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    end_to_end_latency_std_ms: float = 0.0
    gpu_peak_memory_mb: float = 0.0
    fit_time_s: float = 0.0
    warmup_iterations: int = 0
    measured_iterations: int = 0
    device: str = "unknown"
    preprocessing_mode: str = "baseline"
    image_size: int = 224
    batch_size: int = 1
    timestamp: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA if on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int = 10,
    iterations: int = 30,
) -> tuple:
    """Measure the average latency of *fn* in milliseconds.

    Args:
        fn:         Callable that performs the operation to measure.
        device:     Torch device (for CUDA synchronization).
        warmup:     Number of warmup iterations.
        iterations: Number of measured iterations.

    Returns:
        Tuple of (mean_ms, std_ms).
    """
    # Warmup
    for _ in range(warmup):
        fn()
        _synchronize(device)

    # Measured
    times = []
    for _ in range(iterations):
        _synchronize(device)
        t0 = time.perf_counter()
        fn()
        _synchronize(device)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / max(len(times) - 1, 1)) ** 0.5
    return avg, std


def measure_gpu_peak_memory(
    fn: Callable[[], Any],
    device: torch.device,
) -> float:
    """Measure GPU peak memory in MB for a single forward pass.

    Returns 0.0 if not on CUDA.
    """
    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 * 1024)


def profile_model(
    model_only_fn: Callable[[], Any],
    end_to_end_fn: Callable[[], Any],
    device: torch.device,
    warmup: int = 10,
    iterations: int = 30,
    preprocessing_mode: str = "baseline",
    image_size: int = 224,
    config_snapshot: Optional[Dict[str, Any]] = None,
    fit_time_s: float = 0.0,
) -> RuntimeResult:
    """Full profiling: latency + GPU peak memory.

    Args:
        model_only_fn:   Pure inference (no preprocessing, no disk I/O).
        end_to_end_fn:   Preprocessing + H2D + inference + postprocess.
        device:          Torch device.
        warmup:          Warmup iterations (minimum 10).
        iterations:      Measured iterations (minimum 30).
        preprocessing_mode: Name of the preprocessing mode used.
        image_size:      Final image size.
        config_snapshot: Arbitrary config dict to store with results.
        fit_time_s:      Training/fit time in seconds.

    Returns:
        RuntimeResult with all fields populated.

    Raises:
        ValueError: If warmup < 10 or iterations < 30 (protocol violation).
    """
    if warmup < 10:
        raise ValueError(f"Latency protocol requires warmup >= 10, got {warmup}.")
    if iterations < 30:
        raise ValueError(
            f"Latency protocol requires measured iterations >= 30, got {iterations}."
        )
    logger.info(
        "Profiling: warmup=%d  measured=%d  device=%s", warmup, iterations, device
    )

    model_only_ms, model_only_std = measure_latency(
        model_only_fn, device, warmup, iterations
    )
    e2e_ms, e2e_std = measure_latency(end_to_end_fn, device, warmup, iterations)
    gpu_peak = measure_gpu_peak_memory(model_only_fn, device)

    result = RuntimeResult(
        model_only_latency_ms=round(model_only_ms, 3),
        model_only_latency_std_ms=round(model_only_std, 3),
        end_to_end_latency_ms=round(e2e_ms, 3),
        end_to_end_latency_std_ms=round(e2e_std, 3),
        gpu_peak_memory_mb=round(gpu_peak, 2),
        fit_time_s=round(fit_time_s, 2),
        warmup_iterations=warmup,
        measured_iterations=iterations,
        device=str(device),
        preprocessing_mode=preprocessing_mode,
        image_size=image_size,
        batch_size=1,
        timestamp=datetime.now().isoformat(),
        config_snapshot=config_snapshot or {},
    )

    logger.info(
        "Profile: model_only=%.2f ms  e2e=%.2f ms  gpu_peak=%.1f MB",
        result.model_only_latency_ms,
        result.end_to_end_latency_ms,
        result.gpu_peak_memory_mb,
    )
    return result
