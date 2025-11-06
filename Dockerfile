FROM python:3.11-slim

# System basics (fast builds, smaller image)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build tools only if needed by deps (sklearn wheels are prebuilt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source + models
COPY . .

# Health port is provided by Railway in $PORT
CMD ["bash", "-lc", "gunicorn app:app -w 2 -k gthread -b 0.0.0.0:${PORT}"]
