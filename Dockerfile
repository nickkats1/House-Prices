FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for lightgbm/xgboost.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer caching).
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install \
        "pandas" \
        "numpy" \
        "scikit-learn>=1.4" \
        "xgboost" \
        "lightgbm" \
        "joblib" \
        "flask" \
        "gunicorn"

# Project files (artifacts and data are mounted at runtime via compose).
COPY config.py ./
COPY prices/ ./prices/
COPY app.py ./
COPY templates/ ./templates/
COPY static/ ./static/

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "2", "--access-logfile", "-", "app:app"]
