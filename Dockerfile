FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (needed for ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install your project (for src/ imports)
RUN pip install --no-cache-dir -e .

EXPOSE 8080

# Start the app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "application:app"]