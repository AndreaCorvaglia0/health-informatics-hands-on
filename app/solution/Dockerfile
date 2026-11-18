# 1. Base image: small Debian with Python 3.12 (pin the version = reproducible)
FROM python:3.12-slim

# 2. No .pyc files, unbuffered logs (shown immediately in `docker logs`), no pip cache in the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 3. Do not run as root inside the container
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# 4. Install dependencies FIRST: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# 5. Then copy the application code, models and samples (see .dockerignore for what is left out)
COPY --chown=appuser:appuser . .
USER appuser

# 6. Streamlit listens on 8501; the healthcheck calls its built-in health endpoint
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# 7. 0.0.0.0 = reachable from outside the container (the default 'localhost' is not)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
