# Agent Orange embedding worker — Render Background Worker (Standard: 2 GB RAM / 1 CPU).
# Multi-stage: download nomic ONNX + re-quantize vision in the builder; slim runtime
# image has no build tooling. Models are baked in so cold start does not hit Hugging Face.
#
# Build:  docker build -t ao-embed-worker .
# Run:    docker run --rm -e DRY_RUN=1 -e MODEL_PRECISION=int8 ao-embed-worker
#
# LIMIT is intentionally unset here. Unset/empty LIMIT = drain until the queue is
# empty. Set LIMIT (e.g. 500) only for local/test caps — never for production drain.

FROM python:3.12-slim-bookworm AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-build.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-build.txt -r requirements.txt
COPY tools/download_models.py tools/download_models.py
# Pin revs inside download_models.py. BuildKit network required.
RUN python tools/download_models.py /build/models

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    MODEL_PRECISION=int8 \
    ORT_THREADS=1 \
    BATCH_SIZE=50 \
    LOAD_MODE=both \
    IDLE_AFTER_DONE=1 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TUNE=1
# LIMIT is not set: worker drains the queue. For a test run: -e LIMIT=500
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /build/models /app/models
COPY worker.py .
RUN chown -R app:app /app
USER app
# Render Background Workers have no HTTP port. After the queue drains (or LIMIT
# if set for a test), IDLE_AFTER_DONE keeps the process alive so Render does not
# restart; delete/suspend the service when done.
CMD ["python", "-u", "worker.py"]
