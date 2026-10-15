"""
Patch sampling, band normalization, and train/val/test splits for UrbanLens.

Design notes (see project instructions sections 4-5 for the source spec):

- Geographic split: each training city's raster is cut into a west 80% /
  east 20% strip, reserved identically across all of that city's dates. The
  west strip is sampled for train patches, the east strip for val patches.
  This is what keeps the same city block from landing in train in one year
  and val in another.
- Patch budget is per city, divided across that city's available dates —
  Bengaluru's 5 dates don't get 5x the patches of a 3-date city.
- Invalid pixels (S2 nodata, DW nodata, DW Snow/Ice) carry no training
  signal. Candidate patches above `invalid_pixel_frac_max` are rejected
  outright; whatever invalid pixels remain in an accepted patch are set to
  `ignore_index` in the label, excluding them from loss.
- Bare Soil is the thinnest target class (~0.5-1.8% of pixels per the Phase 0
  corpus inspection). A second sampling pass specifically seeks out patches
  containing it rather than trusting uniform random sampling to surface it.
- Band mean/std are computed once from training cities' raw S2 values and
  frozen into configs/config.yaml. z-scoring in raw digital-number units vs.
  true reflectance (DN/10000) gives identical results — (x-mean)/std is
  invariant to a shared scale factor on x, mean, and std — so there's no
  separate unit conversion step here.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from torch.utils.data import Dataset

from spectral_indices import compute_indices

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

FNAME_RE = re.compile(r"^(?P<city>[a-z]+)_(?P<year>\d{4})_(?P<kind>s2|dw)\.tif$")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def city_year_pairs(city):
    years = []
    for p in sorted(DATA_RAW.glob(f"{city}_*_s2.tif")):
        m = FNAME_RE.match(p.name)
        if m and m["city"] == city:
            years.append(int(m["year"]))
    return years


def raster_paths(city, year):
    return DATA_RAW / f"{city}_{year}_s2.tif", DATA_RAW / f"{city}_{year}_dw.tif"


def remap_dw(dw, dw_to_target, ignore_index):
    """DW class ID array -> target class ID array. Anything not in the
    mapping (Snow/Ice=8, any nodata sentinel) becomes ignore_index."""
    out = np.full(dw.shape, ignore_index, dtype=np.uint8)
    for dw_id, target_id in dw_to_target.items():
        out[dw == dw_id] = target_id
    return out


# --------------------------------------------------------------------------
# Band statistics — training cities only
# --------------------------------------------------------------------------

def compute_band_stats(config):
    """Per-band mean/std over valid pixels of every training-city S2 raster.
    Streams one band at a time to keep peak memory low."""
    n_bands = len(config["bands"]["order"])
    total_sum = np.zeros(n_bands, dtype=np.float64)
    total_sumsq = np.zeros(n_bands, dtype=np.float64)
    total_count = np.zeros(n_bands, dtype=np.int64)

    for city in config["cities"]["train"]:
        for year in city_year_pairs(city):
            s2_path, _ = raster_paths(city, year)
            with rasterio.open(s2_path) as src:
                nodata = src.nodatavals[0]
                for b in range(1, n_bands + 1):
                    arr = src.read(b).astype("float64")
                    valid = arr != nodata if nodata is not None else np.ones(arr.shape, bool)
                    v = arr[valid]
                    total_sum[b - 1] += v.sum()
                    total_sumsq[b - 1] += (v ** 2).sum()
                    total_count[b - 1] += v.size
            print(f"  scanned {city}_{year}_s2.tif")

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------
# Patch sampling
# --------------------------------------------------------------------------

class SpacingIndex:
    """Grid-bucketed nearest-neighbor check so enforcing min patch-center
    spacing doesn't degrade to O(n^2) as accepted patches accumulate."""

    def __init__(self, min_spacing):
        self.min_spacing = min_spacing
        self.buckets = defaultdict(list)

    def _bucket(self, row, col):
        return row // self.min_spacing, col // self.min_spacing

    def ok(self, row, col):
        br, bc = self._bucket(row, col)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for r, c in self.buckets[(br + dr, bc + dc)]:
                    if abs(r - row) < self.min_spacing and abs(c - col) < self.min_spacing:
                        return False
        return True

    def add(self, row, col):
        self.buckets[self._bucket(row, col)].append((row, col))


def invalid_frac(s2_win, dw_win, s2_nodata, dw_nodata, snow_ice_id=8):
    invalid = np.zeros(dw_win.shape, dtype=bool)
    if s2_nodata is not None:
        invalid |= np.any(s2_win == s2_nodata, axis=0)
    if dw_nodata is not None:
        invalid |= dw_win == dw_nodata
    invalid |= dw_win == snow_ice_id
    return invalid.mean()


def find_class_pixel_locations(dw_path, class_id, row_range, col_range, patch_size):
    """One full-region read of the DW band, returning every pixel of
    `class_id` a patch could be centered on. Blind uniform random sampling
    essentially never lands on a class that's ~1% of the raster and
    scattered rather than clustered (confirmed empirically: it found <2% of
    its Bare Soil target in 3 of 4 training cities) — searching for actual
    pixel locations first, then centering candidates on them, is what makes
    "oversample thin classes" mean something instead of relying on luck."""
    row_lo, row_hi = row_range
    col_lo, col_hi = col_range
    with rasterio.open(dw_path) as src:
        win = Window(col_lo, row_lo, col_hi - col_lo, row_hi - row_lo)
        arr = src.read(1, window=win)
    rows, cols = np.where(arr == class_id)
    return list(zip((rows + row_lo).tolist(), (cols + col_lo).tolist()))


def sample_patches_in_region(
    s2_path, dw_path, row_range, col_range, n_target, patch_size,
    invalid_frac_max, min_spacing, oversample_target_id=None,
    oversample_frac=0.2, oversample_min_pixels=100, max_attempts_factor=40,
    general_attempts_factor=150, hotspot_jitter=64,
):
    """Random patch-center sampling within [row_range, col_range) of one
    raster, rejecting candidates over the invalid-pixel threshold and below
    the min-spacing distance from already-accepted centers. A first pass
    specifically seeks patches containing `oversample_target_id` (Bare Soil)
    for `oversample_frac` of the budget, centered on real occurrences of that
    class (see find_class_pixel_locations) rather than blind random guessing;
    it runs before the general pass so it isn't fighting an already-packed
    region for the rare slots that satisfy it."""
    accepted = []
    spacing = SpacingIndex(min_spacing)
    row_lo, row_hi = row_range
    col_lo, col_hi = col_range
    if row_hi <= row_lo or col_hi <= col_lo:
        return accepted

    with rasterio.open(s2_path) as s2_src, rasterio.open(dw_path) as dw_src:
        s2_nodata = s2_src.nodatavals[0]
        dw_nodata = dw_src.nodatavals[0]

        def try_accept(row, col, require_class=None):
            row = max(row_lo, min(row, row_hi - patch_size))
            col = max(col_lo, min(col, col_hi - patch_size))
            if not spacing.ok(row, col):
                return False
            win = Window(col, row, patch_size, patch_size)
            dw_win = dw_src.read(1, window=win)
            if require_class is not None:
                if (dw_win == require_class).sum() < oversample_min_pixels:
                    return False
            s2_win = s2_src.read(window=win)
            if invalid_frac(s2_win, dw_win, s2_nodata, dw_nodata) > invalid_frac_max:
                return False
            spacing.add(row, col)
            accepted.append({"s2_path": str(s2_path), "dw_path": str(dw_path),
                              "row": row, "col": col})
            return True

        n_oversample = int(n_target * oversample_frac) if oversample_target_id is not None else 0
        n_general = n_target - n_oversample

        if n_oversample:
            hotspots = find_class_pixel_locations(
                dw_path, oversample_target_id, row_range, col_range, patch_size
            )
            attempts, max_attempts = 0, n_oversample * max_attempts_factor
            while len(accepted) < n_oversample and attempts < max_attempts and hotspots:
                pr, pc = random.choice(hotspots)
                row = pr - patch_size // 2 + random.randint(-hotspot_jitter, hotspot_jitter)
                col = pc - patch_size // 2 + random.randint(-hotspot_jitter, hotspot_jitter)
                try_accept(row, col, require_class=oversample_target_id)
                attempts += 1
            if len(accepted) < n_oversample:
                print(f"  [{Path(s2_path).stem}] oversample pass only found "
                      f"{len(accepted)}/{n_oversample} bare-soil patches "
                      f"({len(hotspots)} candidate pixels in region)")

        start = len(accepted)
        attempts, max_attempts = 0, n_general * general_attempts_factor
        while len(accepted) < start + n_general and attempts < max_attempts:
            row = random.randint(row_lo, row_hi - patch_size)
            col = random.randint(col_lo, col_hi - patch_size)
            try_accept(row, col)
            attempts += 1
        if len(accepted) < start + n_general:
            print(f"  [{Path(s2_path).stem}] general pass only found "
                  f"{len(accepted) - start}/{n_general} patches — region may "
                  f"be near packing capacity at this min_spacing")

    return accepted


def build_patch_index(config, seed=None):
    """Full sampling pass over every training city. Returns
    {'train': [...], 'val': [...]} patch index entries."""
    if seed is not None:
        random.seed(seed)
    patch_cfg = config["patch"]
    patch_size = patch_cfg["size"]
    bare_soil_dw_id = 7  # "Bare ground" in Dynamic World's own schema

    index = {"train": [], "val": []}
    for city in config["cities"]["train"]:
        years = city_year_pairs(city)
        if not years:
            print(f"  !! no rasters found for train city {city}, skipping")
            continue
        s2_probe, _ = raster_paths(city, years[0])
        with rasterio.open(s2_probe) as src:
            height, width = src.height, src.width

        holdout_col_start = int(width * (1 - patch_cfg["holdout_strip_frac"]))
        train_budget = patch_cfg["budget_per_train_city"]
        val_budget = round(train_budget * 0.25)

        per_date_train = train_budget // len(years)
        per_date_val = val_budget // len(years)

        for year in years:
            s2_path, dw_path = raster_paths(city, year)
            train_patches = sample_patches_in_region(
                s2_path, dw_path,
                row_range=(0, height), col_range=(0, holdout_col_start),
                n_target=per_date_train, patch_size=patch_size,
                invalid_frac_max=patch_cfg["invalid_pixel_frac_max"],
                min_spacing=patch_cfg["min_spacing_px"],
                oversample_target_id=bare_soil_dw_id,
            )
            val_patches = sample_patches_in_region(
                s2_path, dw_path,
                row_range=(0, height), col_range=(holdout_col_start, width),
                n_target=per_date_val, patch_size=patch_size,
                invalid_frac_max=patch_cfg["invalid_pixel_frac_max"],
                min_spacing=patch_cfg["min_spacing_px"],
                oversample_target_id=bare_soil_dw_id,
            )
            print(f"[{city} {year}] train={len(train_patches)} val={len(val_patches)}")
            index["train"].extend(train_patches)
            index["val"].extend(val_patches)

    return index


def class_pixel_counts(index, config, sample_n=None):
    """Per-class pixel counts (+ ignored count) over the index, or a random
    sample of it. Used both for reporting coverage and, in train.py, for
    computing loss class weights from the actual training corpus."""
    dw_to_target = config["classes"]["dw_to_target"]
    ignore_index = config["classes"]["ignore_index"]
    names = config["classes"]["names"]
    patch_size = config["patch"]["size"]

    counts = np.zeros(len(names), dtype=np.int64)
    ignored = 0
    sample = index if sample_n is None else random.sample(index, min(sample_n, len(index)))
    for item in sample:
        with rasterio.open(item["dw_path"]) as src:
            win = Window(item["col"], item["row"], patch_size, patch_size)
            dw_win = src.read(1, window=win)
        label = remap_dw(dw_win, dw_to_target, ignore_index)
        ignored += (label == ignore_index).sum()
        for i in range(len(names)):
            counts[i] += (label == i).sum()
    return counts, ignored, len(sample)


def report_class_coverage(index, config, sample_n=200):
    """'Checked rather than assumed' per instructions section 5."""
    names = config["classes"]["names"]
    counts, ignored, n = class_pixel_counts(index, config, sample_n)
    total = counts.sum() + ignored
    print(f"  class coverage over {n} sampled patches:")
    for name, c in zip(names, counts):
        print(f"    {name:<12} {100 * c / total:5.2f}%")
    print(f"    {'ignore':<12} {100 * ignored / total:5.2f}%")


# --------------------------------------------------------------------------
# torch Dataset
# --------------------------------------------------------------------------

class UrbanLensDataset(Dataset):
    def __init__(self, patch_index, config):
        self.patch_index = patch_index
        self.patch_size = config["patch"]["size"]
        self.dw_to_target = config["classes"]["dw_to_target"]
        self.ignore_index = config["classes"]["ignore_index"]

        mean = config["bands"]["mean"]
        std = config["bands"]["std"]
        if mean is None or std is None:
            raise ValueError(
                "config.yaml bands.mean/std are unset — run "
                "`python src/dataset.py --compute-stats` and fill them in first"
            )
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        item = self.patch_index[idx]
        win = Window(item["col"], item["row"], self.patch_size, self.patch_size)

        with rasterio.open(item["s2_path"]) as src:
            s2 = src.read(window=win).astype("float32")
            s2_nodata = src.nodatavals[0]
        with rasterio.open(item["dw_path"]) as src:
            dw = src.read(1, window=win)
            dw_nodata = src.nodatavals[0]

        indices = compute_indices(s2)               # (3, H, W), scale-invariant on raw DN
        s2_norm = (s2 - self.mean) / self.std        # frozen training-city stats
        x = np.concatenate([s2_norm, indices], axis=0).astype("float32")  # (8, H, W)

        label = remap_dw(dw, self.dw_to_target, self.ignore_index)
        if s2_nodata is not None:
            label[np.any(s2 == s2_nodata, axis=0)] = self.ignore_index
        if dw_nodata is not None:
            label[dw == dw_nodata] = self.ignore_index

        return torch.from_numpy(x), torch.from_numpy(label.astype("int64"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-stats", action="store_true",
                     help="compute training-city band mean/std and print them "
                          "(paste into configs/config.yaml by hand)")
    ap.add_argument("--build-index", action="store_true",
                     help="run the full patch sampling pass and save the index")
    args = ap.parse_args()

    config = load_config()

    if args.compute_stats:
        print("Scanning training-city rasters for band statistics...")
        mean, std = compute_band_stats(config)
        print("\nmean:", [round(m, 2) for m in mean])
        print("std: ", [round(s, 2) for s in std])

    if args.build_index:
        index = build_patch_index(config, seed=config.get("seed"))
        print(f"\ntotal: train={len(index['train'])} val={len(index['val'])}")
        report_class_coverage(index["train"], config)

        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out_path = DATA_PROCESSED / "patch_index.json"
        out_path.write_text(json.dumps(index))
        print(f"saved index -> {out_path}")


if __name__ == "__main__":
    main()
