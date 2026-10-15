"""
Phase 7, change detection between two dates of one city (project instructions
section 10). Runs dense tiled inference on both rasters, then derives:

  - transition matrix (km^2) between the 5 classes
  - expansion map: Urban gain / Urban loss
  - transition map: from-class and to-class per pixel
  - a Dynamic World-derived sanity check on the same pair

The model predicts each date independently, so per-date classification noise
shows up as spurious "change". Net Urban area at each date is the most robust
number; gross gain/loss and non-Urban transitions (especially Vegetation <->
Agriculture) carry more noise. Both rasters must share one pixel grid.

Usage:
    python src/change_detection.py --checkpoint outputs/checkpoints/unet5ch_v2_ablation_best.pt \
        --city hyderabad --year-a 2019 --year-b 2024
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from dataset import REPO_ROOT, load_config, raster_paths, remap_dw
from model import UNet
from predict import NODATA, predict_raster

OUT_DIR = REPO_ROOT / "outputs" / "change_detection"
URBAN = 0


def read_dw_target(dw_path, config):
    dw_to_target = config["classes"]["dw_to_target"]
    ignore_index = config["classes"]["ignore_index"]
    with rasterio.open(dw_path) as src:
        dw = src.read(1)
    return remap_dw(dw, dw_to_target, ignore_index).astype(np.uint8)


def transition_matrix(a, b, num_classes):
    """counts[i, j] = pixels of class i at date A and class j at date B,
    ignoring any pixel that is invalid at either date."""
    ok = (a != NODATA) & (b != NODATA) & (a < num_classes) & (b < num_classes)
    idx = a[ok].astype(np.int64) * num_classes + b[ok].astype(np.int64)
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def expansion_map(a, b):
    """1 = Urban gain, 2 = Urban loss, 0 = no Urban change, 255 = invalid."""
    out = np.zeros(a.shape, dtype=np.uint8)
    out[(a != URBAN) & (b == URBAN)] = 1
    out[(a == URBAN) & (b != URBAN)] = 2
    out[(a == NODATA) | (b == NODATA)] = NODATA
    return out


def transition_map(a, b, num_classes):
    """Code = from_class * num_classes + to_class; 255 = invalid."""
    out = (a.astype(np.uint16) * num_classes + b.astype(np.uint16)).astype(np.uint8)
    out[(a == NODATA) | (b == NODATA)] = NODATA
    return out


def summarize(trans, class_names, pixel_km2):
    km2 = trans * pixel_km2
    return {
        "transition_km2": {
            class_names[i]: {class_names[j]: round(float(km2[i, j]), 3)
                              for j in range(len(class_names))}
            for i in range(len(class_names))
        },
        "area_km2_date_a": {class_names[i]: round(float(km2[i].sum()), 3) for i in range(len(class_names))},
        "area_km2_date_b": {class_names[j]: round(float(km2[:, j].sum()), 3) for j in range(len(class_names))},
        "urban_gain_by_source_km2": {
            class_names[i]: round(float(km2[i, URBAN]), 3)
            for i in range(len(class_names)) if i != URBAN
        },
        "urban_gain_km2": round(float(km2[:, URBAN].sum() - km2[URBAN, URBAN]), 3),
        "urban_loss_km2": round(float(km2[URBAN, :].sum() - km2[URBAN, URBAN]), 3),
        "urban_net_change_km2": round(float(km2[:, URBAN].sum() - km2[URBAN, :].sum()), 3),
    }


def write_raster(path, array, profile_src, dtype="uint8"):
    with rasterio.open(profile_src) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype=dtype, nodata=NODATA, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def save_preview(path, exp, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    step = max(1, exp.shape[0] // 1600)
    small = exp[::step, ::step].astype(float)
    shown = np.full(small.shape, np.nan)
    shown[small == 0] = 0
    shown[small == 1] = 1
    shown[small == 2] = 2
    cmap = ListedColormap(["#d9d9d9", "#d7301f", "#2b83ba"])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d9d9d9", label="no Urban change"),
                       Patch(color="#d7301f", label="Urban gain"),
                       Patch(color="#2b83ba", label="Urban loss")],
              loc="lower left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--year-a", type=int, required=True)
    ap.add_argument("--year-b", type=int, required=True)
    ap.add_argument("--in-channels", type=int, default=None, choices=[5, 8])
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tag", type=str, default=None,
                    help="output name tag; defaults to the checkpoint stem")
    args = ap.parse_args()

    config = load_config()
    class_names = config["classes"]["names"]
    num_classes = len(class_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    in_channels = args.in_channels or ckpt["in_channels"]
    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    s2_a, dw_a = raster_paths(args.city, args.year_a)
    s2_b, dw_b = raster_paths(args.city, args.year_b)

    with rasterio.open(s2_a) as sa, rasterio.open(s2_b) as sb:
        if (sa.crs, sa.transform, sa.shape) != (sb.crs, sb.transform, sb.shape):
            raise SystemExit("rasters are not on the same grid; reproject before change detection")
        pixel_km2 = abs(sa.transform.a * sa.transform.e) / 1e6

    print(f"{args.city} {args.year_a} -> {args.year_b}  ({args.checkpoint}, in_channels={in_channels})")
    pred_a, _ = predict_raster(model, s2_a, config, device, args.tile_size, args.batch_size, in_channels)
    pred_b, _ = predict_raster(model, s2_b, config, device, args.tile_size, args.batch_size, in_channels)

    tag = args.tag or Path(args.checkpoint).stem
    pair = f"{args.city}_{args.year_a}_{args.year_b}_{tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_trans = transition_matrix(pred_a, pred_b, num_classes)
    model_exp = expansion_map(pred_a, pred_b)
    write_raster(OUT_DIR / f"{pair}_expansion.tif", model_exp, s2_a)
    write_raster(OUT_DIR / f"{pair}_transition.tif", transition_map(pred_a, pred_b, num_classes), s2_a)
    write_raster(OUT_DIR / f"{pair}_pred_{args.year_a}.tif", pred_a, s2_a)
    write_raster(OUT_DIR / f"{pair}_pred_{args.year_b}.tif", pred_b, s2_a)

    # Dynamic World sanity check: the same statistics derived purely from DW labels
    dw_ta = read_dw_target(dw_a, config)
    dw_tb = read_dw_target(dw_b, config)
    dw_trans = transition_matrix(dw_ta, dw_tb, num_classes)
    dw_exp = expansion_map(dw_ta, dw_tb)

    both = (model_exp != NODATA) & (dw_exp != NODATA)
    m_gain, d_gain = (model_exp == 1) & both, (dw_exp == 1) & both
    inter = int((m_gain & d_gain).sum())
    overlap = {
        "model_gain_pixels": int(m_gain.sum()),
        "dw_gain_pixels": int(d_gain.sum()),
        "gain_agreement_iou": round(inter / max(1, int((m_gain | d_gain).sum())), 4),
        "gain_precision_vs_dw": round(inter / max(1, int(m_gain.sum())), 4),
        "gain_recall_vs_dw": round(inter / max(1, int(d_gain.sum())), 4),
    }

    results = {
        "city": args.city, "year_a": args.year_a, "year_b": args.year_b,
        "checkpoint": args.checkpoint, "in_channels": in_channels,
        "pixel_area_km2": pixel_km2,
        "model": summarize(model_trans, class_names, pixel_km2),
        "dynamic_world": summarize(dw_trans, class_names, pixel_km2),
        "gain_map_overlap_model_vs_dw": overlap,
    }
    (OUT_DIR / f"{pair}_stats.json").write_text(json.dumps(results, indent=2))
    save_preview(OUT_DIR / f"{pair}_expansion.png", model_exp,
                 f"{args.city.title()} Urban change {args.year_a} to {args.year_b} ({tag})")

    m, d = results["model"], results["dynamic_world"]
    print(f"\nUrban area (km^2)      {args.year_a:>10} {args.year_b:>10}   gain    loss     net")
    print(f"  model               {m['area_km2_date_a']['Urban']:>10.1f} {m['area_km2_date_b']['Urban']:>10.1f}"
          f"  {m['urban_gain_km2']:>6.1f}  {m['urban_loss_km2']:>6.1f}  {m['urban_net_change_km2']:>6.1f}")
    print(f"  Dynamic World       {d['area_km2_date_a']['Urban']:>10.1f} {d['area_km2_date_b']['Urban']:>10.1f}"
          f"  {d['urban_gain_km2']:>6.1f}  {d['urban_loss_km2']:>6.1f}  {d['urban_net_change_km2']:>6.1f}")
    print(f"\nUrban-gain map vs Dynamic World: IoU={overlap['gain_agreement_iou']}  "
          f"precision={overlap['gain_precision_vs_dw']}  recall={overlap['gain_recall_vs_dw']}")
    print(f"\nsaved -> {OUT_DIR}/{pair}_*")


if __name__ == "__main__":
    main()
