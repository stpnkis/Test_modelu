# =============================================================================
# RD++ (Revisiting Reverse Distillation) Anomaly Detection - Docker Image
# =============================================================================
# Base:    pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
#          → Python 3.10, torch==2.1.0, torchvision==0.16.0, CUDA 11.8
# Tested:  NVIDIA GTX 1080, driver ≥ 520, host CUDA ≥ 11.8
#
# Reproduce on a new machine:
#   docker context use default          # must NOT use Docker Desktop context
#   docker compose build
#   docker compose run --rm rd_plus_plus python src/experiment_runner.py
# =============================================================================

# --platform locks the image to amd64 — avoids accidental arm64 pulls on
# Apple Silicon or multi-arch hosts where the CUDA image may not exist.
FROM --platform=linux/amd64 pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Flush Python output immediately (no buffering) — cleaner container logs
ENV PYTHONUNBUFFERED=1
# Prevent .pyc files cluttering the image layers
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies required by Pillow / OpenCV (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pin pip itself so the resolver behaves identically across machines
RUN pip install --no-cache-dir "pip==24.3.1"

# Install Python dependencies before copying source (Docker layer cache:
# requirements rarely change, code changes often → faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code (live-edited via docker-compose volume mount)
COPY src/ src/
COPY tests/ tests/

# Default: open a bash shell; override with `docker compose run rd_plus_plus python ...`
CMD ["bash"]
