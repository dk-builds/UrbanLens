"""
Phase 5, held-out city metrics (project instructions section 8, primary
metric). Dense tiled inference over the *entire* raster of each held-out
city/year — not a patch sample, since the reported number is a generalization
claim about the whole city, not about a subset of it.

Held-out cities contribute zero training patches (see dataset.py's
`cities.train` vs `cities.test` split), so this measures unseen geography
throughout; any held-out city with more than one exported year additionally
covers unseen-year generalization for those extra dates (pooled per-city
report below, not just per city-year).

This scores model predictions against Dynamic World, i.e. agreement with the
training signal's own pseudo-labels — not ground truth. See the project
instructions section 8: independent 200-300 point hand-labeled validation
against both the model and Dynamic World is a separate, required step, not
yet implemented here.

Usage:
    python src/evaluate.py --checkpoint outputs/checkpoints/unet8ch_best.pt
    python src/evaluate.py --checkpoint outputs/checkpoints/unet5ch_ablation_best.pt --in-channels 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from dataset import (
    REPO_ROOT, city_year_pairs, load_config, raster_paths, remap_dw,
)
from model import UNet
from predict import build_input, iter_tile_origins
from train import confusion_matrix, iou_from_confusion

METRICS_DIR = REPO_ROOT / "outputs" / "metrics"

# Value used to fill DW reads past the raster edge (boundless tiles at the
# right/bottom border). Any value outside dw_to_target's keys (0-7) resolves
# to ignore_index via remap_dw's default fill, so this just needs to not
# collide with a real class id — 255 matches the convention already used for
# DW's own explicit nodata sentinel elsewhere in the corpus.
DW_PAD_FILL = 255


def precision_recall_f1(cm):
    """Per-class P/R/F1 from a confusion matrix; NaN for classes absent from
    both preds and labels, matching iou_from_confusion's convention."""
    cm = cm.astype("float64")
    tp = np.diag(cm)
    pred_pos = cm.sum(0)
    actual_pos = cm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(pred_pos > 0, tp / pred_pos, np.nan)
        recall = np.where(actual_pos > 0, tp / actual_pos, np.nan)
        f1 = np.where(precision + recall > 0,
                       2 * precision * recall / (precision + recall), np.nan)
    return precision, recall, f1


@torch.no_grad()
def evaluate_raster(model, s2_path, dw_path, config, device, tile_size, batch_size, in_channels):
    dw_to_target = config["classes"]["dw_to_target"]
    ignore_index = config["classes"]["ignore_index"]
    num_classes = len(config["classes"]["names"])
    mean = np.array(config["bands"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(config["bands"]["std"], dtype=np.float32).reshape(-1, 1, 1)

    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with rasterio.open(s2_path) as s2_src, rasterio.open(dw_path) as dw_src:
        height, width = s2_src.height, s2_src.width
        origins = list(iter_tile_origins(height, width, tile_size))

        for i in range(0, len(origins), batch_size):
            batch_origins = origins[i:i + batch_size]
            xs, labels = [], []
            for row, col in batch_origins:
                win = Window(col, row, tile_size, tile_size)
                # boundless: right/bottom-edge tiles past the raster extent
                # are padded rather than shrunk, since the model needs
                # spatial dims divisible by 16 (4 pooling levels).
                s2 = s2_src.read(window=win, boundless=True, fill_value=0).astype("float32")
                dw = dw_src.read(1, window=win, boundless=True, fill_value=DW_PAD_FILL)
                label = remap_dw(dw, dw_to_target, ignore_index)
                xs.append(build_input(s2, mean, std, in_channels))
                labels.append(label)

            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(labels).astype("int64")).to(device)
            preds = model(x).argmax(dim=1)
            cm_total += confusion_matrix(preds.flatten(), y.flatten(), num_classes, ignore_index).cpu()

    return cm_total.numpy()


def report(cm, class_names, label, exclude_label_class=None):
    """Metrics from a confusion matrix (rows = DW label, cols = prediction).
    exclude_label_class drops every pixel whose DW label is that class (used
    for a land-only view that removes open sea, which is trivially easy and
    inflates coastal cities' scores). The excluded class is left out of mIoU;
    predictions of it on land pixels still count as errors for the land classes."""
    cm = cm.copy()
    skip = None
    if exclude_label_class is not None:
        skip = class_names.index(exclude_label_class)
        cm[skip, :] = 0
    iou = iou_from_confusion(cm)
    precision, recall, f1 = precision_recall_f1(cm)
    if skip is not None:
        iou[skip] = f1[skip] = precision[skip] = recall[skip] = np.nan
    miou = float(np.nanmean(iou))
    print(f"\n{label}  (mIoU={miou:.4f})")
    print(f"  {'class':<12} {'IoU':>7} {'F1':>7} {'precision':>10} {'recall':>8} {'n_pixels':>12}")
    for i, name in enumerate(class_names):
        n = int(cm[i].sum())
        iou_s = f"{iou[i]:.4f}" if not np.isnan(iou[i]) else "  NaN"
        f1_s = f"{f1[i]:.4f}" if not np.isnan(f1[i]) else "  NaN"
        p_s = f"{precision[i]:.4f}" if not np.isnan(precision[i]) else "  NaN"
        r_s = f"{recall[i]:.4f}" if not np.isnan(recall[i]) else "  NaN"
        print(f"  {name:<12} {iou_s:>7} {f1_s:>7} {p_s:>10} {r_s:>8} {n:>12}")
    return {
        "miou": miou,
        "per_class": {
            name: {
                "iou": None if np.isnan(iou[i]) else float(iou[i]),
                "f1": None if np.isnan(f1[i]) else float(f1[i]),
                "precision": None if np.isnan(precision[i]) else float(precision[i]),
                "recall": None if np.isnan(recall[i]) else float(recall[i]),
                "n_pixels": int(cm[i].sum()),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }


def merge_vegetation_agriculture(cm, class_names):
    """Merge Vegetation + Agriculture into one 'Green' class (reporting view
    only; training still uses 5 classes). Returns (merged cm, merged names)."""
    veg, agri = class_names.index("Vegetation"), class_names.index("Agriculture")
    keep = [i for i in range(len(class_names)) if i not in (veg, agri)]
    groups = [[i] for i in keep[:1]] + [[veg, agri]] + [[i] for i in keep[1:]]
    names = [class_names[g[0]] if len(g) == 1 else "Vegetation+Agriculture" for g in groups]
    merged = np.zeros((len(groups), len(groups)), dtype=cm.dtype)
    for a, rows in enumerate(groups):
        for b, cols in enumerate(groups):
            merged[a, b] = cm[np.ix_(rows, cols)].sum()
    return merged, names


def report_both(cm, class_names, label):
    out = report(cm, class_names, label)
    out["land_only"] = report(cm, class_names, f"{label} [land only, DW Water pixels excluded]",
                               exclude_label_class="Water")
    mcm, mnames = merge_vegetation_agriculture(cm, class_names)
    out["land_only_merged_veg_agri"] = report(
        mcm, mnames, f"{label} [land only, Vegetation+Agriculture merged]",
        exclude_label_class="Water")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--in-channels", type=int, default=None, choices=[5, 8],
                     help="defaults to the value stored in the checkpoint")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--run-name", type=str, default=None,
                     help="output filename stem; defaults to the checkpoint's stem")
    args = ap.parse_args()

    config = load_config()
    class_names = config["classes"]["names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    in_channels = args.in_channels or ckpt["in_channels"]
    model = UNet(in_channels=in_channels, num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {args.checkpoint} (in_channels={in_channels}, "
          f"trained to epoch {ckpt.get('epoch')}, train val_miou={ckpt.get('val_miou')})")

    results = {"checkpoint": args.checkpoint, "in_channels": in_channels, "cities": {}}
    cm_all = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for city in config["cities"]["test"]:
        years = city_year_pairs(city)
        cm_city = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for year in years:
            s2_path, dw_path = raster_paths(city, year)
            cm = evaluate_raster(model, s2_path, dw_path, config, device,
                                  args.tile_size, args.batch_size, in_channels)
            results["cities"][f"{city}_{year}"] = report_both(cm, class_names, f"{city} {year}")
            cm_all += cm
            cm_city += cm
        if len(years) > 1:
            years_label = "+".join(str(y) for y in years)
            results[f"{city}_pooled_{years_label}"] = report_both(
                cm_city, class_names, f"{city} ({years_label} pooled)")

    results["overall_pooled"] = report_both(cm_all, class_names, "ALL HELD-OUT CITIES (pooled)")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or Path(args.checkpoint).stem
    out_path = METRICS_DIR / f"eval_{run_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
