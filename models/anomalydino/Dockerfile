# =============================================================================
# AnomalyDINO Anomaly Detection — Docker Image
# =============================================================================
# Base:    pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
#          → Python 3.10, torch==2.1.0, torchvision==0.16.0, CUDA 11.8
# Tested:  NVIDIA GTX 1080, driver ≥ 520, host CUDA ≥ 11.8
#
# Reproduce:
#   docker compose build
#   docker compose run --rm anomalydino python src/run_benchmark.py
#
# Jetson Orin Nano:
#   Replace the FROM line with the appropriate L4T PyTorch base image:
#     FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3
#   Then rebuild.  The rest of the Dockerfile stays the same.
# =============================================================================

FROM --platform=linux/amd64 pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps for Pillow / image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pin pip for reproducible resolves
RUN pip install --no-cache-dir "pip==24.3.1"

# Python deps (layer-cached — changes only when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code (overridden by volume mount during development)
COPY src/ src/

CMD ["bash"]
