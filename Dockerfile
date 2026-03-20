# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/front-end
COPY front-end/package*.json ./
RUN npm ci
COPY front-end/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — installed at build time, not at startup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source
COPY app.py download_model.py server.py ./

# Copy built React from Stage 1
COPY --from=frontend-builder /app/front-end/dist ./front-end/dist

# Models volume — mount here to persist across container restarts
VOLUME ["/app/models"]
ENV HF_HOME=/app/models/hf_cache

EXPOSE 8000

CMD ["python", "app.py"]
