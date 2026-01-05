# --------- Stage 1: builder (installs deps + builds wheels) ----------
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build dependencies for packages that need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps into a separate directory (/install)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy project and install it (non-editable for production)
COPY . .
RUN pip install --no-cache-dir --prefix=/install .


# --------- Stage 2: runtime (small image) ----------
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Only runtime libs needed (no gcc/build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy app source (templates/static/etc.)
COPY . .

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "180", "--workers", "2", "--threads", "2", "application:app"]