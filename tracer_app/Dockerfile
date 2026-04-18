# ── Tracer — Production Dockerfile ────────────────────────────────────────────
# Lightweight Python image with system deps required by OpenCV / PyWavelets
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create uploads directory
RUN mkdir -p uploads

EXPOSE 10000

CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "120"]
