# Use lightweight official Python image
FROM --platform=linux/amd64 python:3.10-slim

# Install system dependencies (FFmpeg for torchcodec/audio decoding, libsndfile1 for soundfile)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install uv, then use it to install Python dependencies
RUN pip install uv && \
    uv venv --clear && \
    uv pip install --no-cache --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    # Ensure latest google-genai which includes types.Part.from_bytes
    uv pip install --no-cache "google-genai>=0.3.0" && \
    # Install torchcodec explicitly for audio decoding backend
    uv pip install --no-cache torchcodec==0.5

ENV PATH="/app/.venv/bin:$PATH"

# Ensure project root is on PYTHONPATH so `normalizer` and other local modules are importable
ENV PYTHONPATH=/app

# Default command now runs evaluation automatically based on env vars (DATASET_PATH, DATASET, SPLIT, MODEL_NAME)
CMD ["/bin/sh", "-c", "python run_eval.py --dataset_path $DATASET_PATH --dataset $DATASET --split $SPLIT --model_name $MODEL_NAME --max_samples 1"] 