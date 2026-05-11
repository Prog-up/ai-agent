# ── Base: official OpenVINO Python runtime ────────────────────────────────────
# Use the latest openvino/ubuntu22_runtime which includes OpenVINO >= 2026.x
FROM openvino/ubuntu22_runtime:2026.1.0

USER root

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# Install the custom optimum-intel branch that supports Gemma 4
# Pin to exact commit for reproducibility — update intentionally when upgrading
ARG OPTIMUM_INTEL_SHA=eac389347523177511abe37908090d9e5c12e714
RUN pip3 install --no-cache-dir \
    "git+https://github.com/rkazants/optimum-intel.git@${OPTIMUM_INTEL_SHA}" \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir \
    transformers==5.5.0 \
    torchvision Pillow requests \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir \
    fastapi uvicorn[standard] pydantic prometheus-fastapi-instrumentator

# ── Server code ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY server.py /app/server.py

# Model is mounted at runtime — not baked into the image
VOLUME ["/models"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -sf http://localhost:8000/health || exit 1

# Drop to non-root
RUN useradd -m -u 1001 ovuser
USER ovuser

CMD ["python3", "/app/server.py"]
