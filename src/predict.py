"""
Sliding-window inference on any 5-band Sentinel-2 GeoTIFF, city-agnostic
(project instructions section 13). Shared by evaluate.py, change_detection.py
and the API, and kept free of training dependencies (no mlflow) so the API
image stays small.

Input: bands in config order (B02, B03, B04, B08, B11), L2A surface
reflectance on the 0-10000 scale, ideally 10 m. Normalization uses the frozen
training-city band stats in configs/config.yaml; the 8-channel model also gets
NDVI/NDBI/MNDWI through the same spectral_indices code path as training.

Usage:
    python src/predict.py --checkpoint outputs/checkpoints/unet5ch_v2_ablation_best.pt \
        --input data/raw/kochi_2024_s2.tif --output kochi_2024_classes.tif
"""

import argparse

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from dataset import load_config
from model import UNet
from spectral_indices import compute_indices

NODATA = 255

# Same colours as the site's land-cover layer, so the GeoTIFF renders alike in QGIS.
CLASS_COLORS = {0: (232, 74, 95), 1: (23, 74, 45), 2: (107, 138, 189), 3: (87, 188, 136), 4: (176, 124, 198)}


def load_model(checkpoint, num_classes, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model = UNet(in_channels=ckpt["in_channels"], num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["in_channels"]


def build_input(s2_tile, mean, std, in_channels):
    """Mirrors UrbanLensDataset.__getitem__'s normalization exactly, so
    evaluation sees the same feature space training did."""
    s2_norm = (s2_tile - mean) / std
    if in_channels == 8:
        indices = compute_indices(s2_tile)
        return np.concatenate([s2_norm, indices], axis=0).astype("float32")
    return s2_norm.astype("float32")


def iter_tile_origins(height, width, tile_size):
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            yield row, col


@torch.no_grad()
def predict_raster(model, s2_path, config, device, tile_size, batch_size, in_channels):
    """Full-raster class prediction (uint8) and validity mask (bool)."""
    mean = np.array(config["bands"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(config["bands"]["std"], dtype=np.float32).reshape(-1, 1, 1)

    with rasterio.open(s2_path) as src:
        height, width = src.height, src.width
        pred = np.full((height, width), NODATA, dtype=np.uint8)
        valid = np.zeros((height, width), dtype=bool)
        origins = list(iter_tile_origins(height, width, tile_size))

        for i in range(0, len(origins), batch_size):
            batch = origins[i:i + batch_size]
            xs, raws = [], []
            for row, col in batch:
                win = Window(col, row, tile_size, tile_size)
                s2 = src.read(window=win, boundless=True, fill_value=0).astype("float32")
                raws.append(s2)
                xs.append(build_input(s2, mean, std, in_channels))
            x = torch.from_numpy(np.stack(xs)).to(device)
            out = model(x).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for (row, col), p, raw in zip(batch, out, raws):
                h = min(tile_size, height - row)
                w = min(tile_size, width - col)
                # a pixel with no positive reflectance in any band is no-data
                tile_valid = raw.max(axis=0)[:h, :w] > 0
                pred[row:row + h, col:col + w] = np.where(tile_valid, p[:h, :w], NODATA)
                valid[row:row + h, col:col + w] = tile_valid
    return pred, valid


def write_classes(dst_path, pred, src_path, class_names):
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=NODATA, compress="lzw")
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(pred, 1)
        dst.write_colormap(1, CLASS_COLORS)
        dst.update_tags(classes=",".join(f"{i}={n}" for i, n in enumerate(class_names)), nodata=str(NODATA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config = load_config()
    class_names = config["classes"]["names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, in_channels = load_model(args.checkpoint, len(class_names), device)
    pred, _ = predict_raster(model, args.input, config, device, args.tile_size, args.batch_size, in_channels)
    write_classes(args.output, pred, args.input, class_names)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
