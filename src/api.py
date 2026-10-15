"""
FastAPI service (project instructions section 12). Upload a 5-band Sentinel-2
GeoTIFF, get back the land-cover classes as a GeoTIFF on the same grid, or a
JSON area summary.

Run locally:
    cd src && uvicorn api:app --port 8000
    curl -F file=@../examples/kochi_2024_sample.tif localhost:8000/classify -o classes.tif

Settings (environment variables):
    URBANLENS_CHECKPOINT   model checkpoint (default: the 5-band model)
    URBANLENS_MAX_PIXELS   largest raster accepted, in pixels (default 25M, ~50x50 km at 10 m)
    URBANLENS_MAX_UPLOAD_MB  largest upload accepted (default 300)
    URBANLENS_THREADS      torch CPU threads (default: torch's choice)
"""

import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import rasterio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response
from rasterio.errors import RasterioIOError

from dataset import REPO_ROOT, load_config
from predict import NODATA, load_model, predict_raster, write_classes

CHECKPOINT = Path(os.environ.get("URBANLENS_CHECKPOINT",
                                 REPO_ROOT / "outputs" / "checkpoints" / "unet5ch_v2_ablation_best.pt"))
MAX_PIXELS = int(os.environ.get("URBANLENS_MAX_PIXELS", 25_000_000))
MAX_UPLOAD_BYTES = int(os.environ.get("URBANLENS_MAX_UPLOAD_MB", 300)) * 1024 * 1024
TILE_SIZE = 256
BATCH_SIZE = 8

state = {}
# One inference at a time: each run holds a full-raster array, so this bounds memory.
infer_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    if os.environ.get("URBANLENS_THREADS"):
        torch.set_num_threads(int(os.environ["URBANLENS_THREADS"]))
    if not CHECKPOINT.exists():
        raise RuntimeError(f"checkpoint not found: {CHECKPOINT} (set URBANLENS_CHECKPOINT)")
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, in_channels = load_model(CHECKPOINT, len(config["classes"]["names"]), device)
    state.update(config=config, model=model, in_channels=in_channels, device=device)
    yield
    state.clear()


app = FastAPI(
    title="UrbanLens",
    description="Land-cover segmentation (Urban, Vegetation, Water, Agriculture, Bare Soil) "
                "of 10 m Sentinel-2 imagery over Indian cities.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "device": state.get("device")}


@app.get("/model")
def model_info():
    config = state["config"]
    return {
        "checkpoint": CHECKPOINT.name,
        "in_channels": state["in_channels"],
        "classes": {i: n for i, n in enumerate(config["classes"]["names"])},
        "nodata": NODATA,
        "input": {
            "bands": config["bands"]["order"],
            "scale": "Sentinel-2 L2A surface reflectance, 0-10000 (COPERNICUS/S2_SR_HARMONIZED)",
            "resolution_m": 10,
            "max_pixels": MAX_PIXELS,
        },
    }


def save_upload(upload, folder):
    path = Path(folder) / "input.tif"
    size = 0
    with open(path, "wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"upload larger than {MAX_UPLOAD_BYTES // 2**20} MB")
            f.write(chunk)
    return path


def check_input(path, n_bands):
    """Rejects what the model cannot read; returns warnings for what it can
    read but was not trained on."""
    try:
        src = rasterio.open(path)
    except RasterioIOError:
        raise HTTPException(400, "not a readable raster; upload a GeoTIFF")
    with src:
        if src.count != n_bands:
            raise HTTPException(422, f"expected {n_bands} bands (B02, B03, B04, B08, B11), got {src.count}")
        if src.width * src.height > MAX_PIXELS:
            raise HTTPException(413, f"raster is {src.width}x{src.height}; limit is {MAX_PIXELS:,} pixels")
        warnings = []
        if src.crs is None:
            warnings.append("no CRS; output will not be georeferenced")
        elif src.crs.is_projected and src.crs.linear_units in ("metre", "meter"):
            rx, ry = src.res
            if abs(rx - 10) > 1 or abs(ry - 10) > 1:
                warnings.append(f"pixel size is {rx:g} x {ry:g} m; the model was trained on 10 m pixels")
        else:
            warnings.append("CRS is not projected in metres; cannot check the 10 m pixel size, and areas are not computed")
        step = max(1, max(src.width, src.height) // 512)
        sample = src.read(1, out_shape=(max(1, src.height // step), max(1, src.width // step)))
        sample = sample[sample > 0]
        if sample.size and np.median(sample) < 2:
            warnings.append("blue band median is below 2; values look like 0-1 reflectance, expected 0-10000")
    return warnings


def classify(upload):
    config = state["config"]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = save_upload(upload, tmp)
        warnings = check_input(in_path, len(config["bands"]["order"]))
        with infer_lock:
            pred, _ = predict_raster(state["model"], in_path, config, state["device"],
                                     TILE_SIZE, BATCH_SIZE, state["in_channels"])
        out_path = Path(tmp) / "classes.tif"
        write_classes(out_path, pred, in_path, config["classes"]["names"])
        with rasterio.open(in_path) as src:
            pixel_m2 = abs(src.transform.a * src.transform.e) if src.crs and src.crs.is_projected else None
        return out_path.read_bytes(), pred, pixel_m2, warnings


@app.post("/classify", response_class=Response,
          responses={200: {"content": {"image/tiff": {}}, "description": "uint8 class raster, 255 = no data"}})
def classify_geotiff(file: UploadFile = File(..., description="5-band Sentinel-2 GeoTIFF")):
    """Returns the class raster as a GeoTIFF on the input's grid, with a colour table."""
    data, _, _, warnings = classify(file)
    headers = {"Content-Disposition": 'attachment; filename="urbanlens_classes.tif"'}
    if warnings:
        headers["X-UrbanLens-Warnings"] = json.dumps(warnings)
    return Response(content=data, media_type="image/tiff", headers=headers)


@app.post("/classify/summary")
def classify_summary(file: UploadFile = File(..., description="5-band Sentinel-2 GeoTIFF")):
    """Returns class shares, and areas in km^2 when the input CRS is in metres."""
    _, pred, pixel_m2, warnings = classify(file)
    names = state["config"]["classes"]["names"]
    counts = np.bincount(pred[pred != NODATA], minlength=len(names))
    valid = int(counts.sum())
    classes = {}
    for i, name in enumerate(names):
        classes[name] = {
            "pixels": int(counts[i]),
            "fraction": round(counts[i] / valid, 4) if valid else None,
            "area_km2": round(counts[i] * pixel_m2 / 1e6, 3) if pixel_m2 else None,
        }
    return {"width": int(pred.shape[1]), "height": int(pred.shape[0]), "valid_pixels": valid,
            "nodata_pixels": int(pred.size - valid), "classes": classes, "warnings": warnings}
