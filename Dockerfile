FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps (kept minimal; many wheels ship prebuilt binaries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the repository
COPY . /app

# Default command (edit config.yaml to switch modes)
CMD ["python3", "main.py"]

