FROM python:3.12-slim

WORKDIR /app

# Force CPU Torch

# Copy project
COPY . .

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies into system python (no venv needed)
RUN uv pip install --system .


CMD ["uv", "run", "main.py"]