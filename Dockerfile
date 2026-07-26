# Reproducible container for building and deploying UniDocShield
FROM python:3.10-slim

# Prevent Python from writing pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ /app/src/
COPY run_pipeline.py /app/

# Expose ports: 8000 for FastAPI API, 9090 for prometheus client (or metrics mapped on uvicorn)
EXPOSE 8000

# Run uvicorn server in production mode
CMD ["python", "-m", "uvicorn", "src.production.app:app", "--host", "0.0.0.0", "--port", "8000"]
