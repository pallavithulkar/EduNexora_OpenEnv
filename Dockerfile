# EduNexora AI — Dockerfile
# HF Spaces compatible (port 7860)

FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────── #
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF Spaces requirement) ─────────────────────────────────── #
RUN useradd -m -u 1000 appuser
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────── #
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir openai pydantic flask pandas pdfplumber Werkzeug gunicorn

# ── Application files ─────────────────────────────────────────────────────── #
COPY . .

# ── Uploads directory ─────────────────────────────────────────────────────── #
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app

USER appuser

# ── Environment defaults ──────────────────────────────────────────────────── #
ENV API_BASE_URL="https://api-inference.huggingface.co/v1/" \
    MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
    HF_TOKEN="" \
    FLASK_ENV="production" \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────── #
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Start ─────────────────────────────────────────────────────────────────── #
# SAFE VERSION: Using python directly instead of gunicorn to ensure 
# the inference logs (if __name__ == "__main__":) execute properly.
CMD ["python", "app.py"]