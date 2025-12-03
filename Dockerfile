# Hand-of-Sauron AR Docker Configuration
# Multi-stage build for minimal final image size

FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port for potential web streaming (optional)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2; print('OpenCV available')" || exit 1

# Set entrypoint - default to headless mode for containerized environments
ENTRYPOINT ["python", "headless_runner.py"]

# Alternative entrypoints (uncomment to use different modes):
# Full AR mode: ENTRYPOINT ["python", "handDanger.py"]
# Demo mode: ENTRYPOINT ["python", "demo_mode.py"]  
# Auto-detection: ENTRYPOINT ["python", "fix_and_run.py"]