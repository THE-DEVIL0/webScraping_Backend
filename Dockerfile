FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is installed
RUN python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT