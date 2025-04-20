FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for better performance
RUN pip install --no-cache-dir xformers triton

# Enable write permissions for saving checkpoints
RUN chmod -R 777 /workspace

# Set environment variables
ENV PYTHONPATH=/workspace
ENV TORCH_HOME=/workspace/.torch
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV HF_HOME=/workspace/.cache/huggingface

# Default command
CMD ["bash"] 