# syntax=docker/dockerfile:1

# Slim Python base — small, official, well-maintained.
# Pin to 3.11 to match your uv project (avoids torch wheel surprises).
FROM python:3.11-slim

# System deps: libgomp1 for numpy/torch, libjpeg + libpng for Pillow.
# --no-install-recommends keeps the image lean; rm apt lists at the end.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libjpeg62-turbo \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps FIRST, in their own layer.
# If you edit code but not requirements.txt, Docker reuses this cached layer
# on rebuild — dependency install goes from ~5min to ~1sec.
COPY requirements.txt .

# PyTorch defaults to CUDA wheels on Linux (~2GB of GPU libraries we don't need).
# Install CPU-only torch from PyTorch's dedicated index first.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0

# Then install everything else, stripping out CUDA-related packages
# (they'd try to pull GPU torch back in, defeating the previous step).
RUN grep -v -E "^(nvidia-|cuda-|triton==|torch==|torchvision==)" requirements.txt > requirements-cpu.txt \
    && pip install --no-cache-dir -r requirements-cpu.txt

# Then copy code — only what the API needs at runtime.
COPY src/ ./src/
COPY api/ ./api/
COPY config.yaml .

# The checkpoint goes in via COPY as well.
# Alternative would be a bind mount at runtime; baking it in makes the image
# fully self-contained — one `docker run` and it works.
COPY checkpoints/best.pt ./checkpoints/best.pt

# Document the port the container listens on (metadata only — doesn't publish).
EXPOSE 8000

# Basic container-level health check so Docker can report container health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Bind to 0.0.0.0 (all interfaces) so the port is reachable from outside
# the container. 127.0.0.1 would only accept connections from inside.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]