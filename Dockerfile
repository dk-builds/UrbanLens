# UrbanLens API, CPU-only. The rasterio wheel ships its own GDAL; on a slim
# Python base it only needs libexpat from the system.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    URBANLENS_CHECKPOINT=/models/model.pt

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && pip install -r requirements-api.txt

COPY configs/config.yaml configs/
COPY src/api.py src/predict.py src/model.py src/spectral_indices.py src/dataset.py src/

RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /app/src
EXPOSE 8000

# The checkpoint is not baked in: mount it read-only at /models/model.pt.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
