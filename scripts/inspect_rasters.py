#!/usr/bin/env python3
"""
Inspect the UrbanLens raster corpus before any dataset code gets written.

Answers three questions:
  1. Are all the Sentinel-2 scenes the same processing level (L2A/BOA vs L1C/TOA)?
  2. Are same-city rasters pixel-aligned across years? (change detection needs this)
  3. What season does each scene look like, inferred from Dynamic World class mix?

Usage:
    python inspect_rasters.py /path/to/data/raw
    python inspect_rasters.py /path/to/data/raw --max-dim 4000
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

# Dynamic World class IDs as exported by GEE (the 'label' band).
DW_CLASSES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_veg",
    4: "crops",
    5: "shrub_scrub",
    6: "built",
    7: "bare",
    8: "snow_ice",
}

# S2 band order per the project spec: B02, B03, B04, B08, B11.
S2_BANDS = ["B02_blue", "B03_green", "B04_red", "B08_nir", "B11_swir"]

FNAME_RE = re.compile(r"^(?P<city>[a-z]+)_(?P<year>\d{4})_(?P<kind>s2|dw)\.tif$", re.I)


def parse_name(path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    d["year"] = int(d["year"])
    d["kind"] = d["kind"].lower()
    d["city"] = d["city"].lower()
    d["path"] = path
    return d


def read_downsampled(src, band, max_dim):
    """Read one band, downsampled so the long edge is <= max_dim."""
    scale = max(src.width, src.height) / max_dim
    if scale <= 1:
        out_h, out_w = src.height, src.width
    else:
        out_h, out_w = int(src.height / scale), int(src.width / scale)
    return src.read(
        band, out_shape=(out_h, out_w), resampling=Resampling.nearest
    ), (out_h, out_w)


def valid_mask(arr, nodata):
    m = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, bool)
    if nodata is not None:
        m &= arr != nodata
    return m


def inspect_s2(rec, max_dim):
    """Per-band stats + a heuristic read on processing level."""
    out = {}
    with rasterio.open(rec["path"]) as src:
        out["shape"] = (src.height, src.width)
        out["count"] = src.count
        out["crs"] = str(src.crs)
        out["transform"] = tuple(round(v, 4) for v in src.transform[:6])
        out["dtype"] = src.dtypes[0]
        out["nodata"] = src.nodatavals[0]
        out["res"] = (abs(src.transform.a), abs(src.transform.e))
        out["tags"] = {k: v for k, v in src.tags().items() if len(str(v)) < 200}

        bands = []
        blue_p5 = blue_med = red_med = None
        for b in range(1, src.count + 1):
            arr, _ = read_downsampled(src, b, max_dim)
            m = valid_mask(arr, src.nodatavals[b - 1])
            v = arr[m].astype("float64")
            if v.size == 0:
                bands.append(None)
                continue
            stats = {
                "p1": float(np.percentile(v, 1)),
                "p5": float(np.percentile(v, 5)),
                "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)),
                "max": float(v.max()),
                "valid_frac": float(m.mean()),
            }
            bands.append(stats)
            if b == 1:
                blue_p5, blue_med = stats["p5"], stats["median"]
            if b == 3:
                red_med = stats["median"]
        out["bands"] = bands

    # Heuristic. Atmospheric path radiance puts a floor under the blue band in
    # top-of-atmosphere data; surface-reflectance products let dark pixels go
    # near zero. Also, TOA blue usually sits above TOA red over land, while BOA
    # blue sits below. Neither test is proof -- confirm against the GEE script.
    verdict, why = "unknown", []
    if blue_p5 is not None:
        scale = 10000.0 if out["bands"][0]["p95"] > 1.5 else 1.0
        bp5, bmed, rmed = blue_p5 / scale, blue_med / scale, (red_med or 0) / scale
        why.append(f"blue p5={bp5:.4f}, blue median={bmed:.4f}, red median={rmed:.4f}")
        toa_votes = int(bp5 > 0.06) + int(bmed > rmed)
        boa_votes = int(bp5 < 0.03) + int(bmed < rmed)
        if toa_votes >= 2:
            verdict = "likely TOA / L1C"
        elif boa_votes >= 2:
            verdict = "likely BOA / L2A"
        else:
            verdict = "ambiguous"
    out["level"] = verdict
    out["level_why"] = why
    return out


def inspect_dw(rec, max_dim):
    out = {}
    with rasterio.open(rec["path"]) as src:
        out["shape"] = (src.height, src.width)
        out["count"] = src.count
        out["crs"] = str(src.crs)
        out["transform"] = tuple(round(v, 4) for v in src.transform[:6])
        out["dtype"] = src.dtypes[0]
        out["nodata"] = src.nodatavals[0]
        out["res"] = (abs(src.transform.a), abs(src.transform.e))
        arr, _ = read_downsampled(src, 1, max_dim)
        m = valid_mask(arr, src.nodatavals[0])
        v = arr[m]
        out["valid_frac"] = float(m.mean())
        vals, counts = np.unique(v, return_counts=True)
        total = counts.sum()
        out["hist"] = {
            int(k): float(c / total) for k, c in zip(vals, counts)
        }
        out["unexpected"] = [int(k) for k in vals if int(k) not in DW_CLASSES]
    return out


def fmt_pct(x):
    return f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--max-dim", type=int, default=3000,
                    help="downsample long edge to this many px for stats (default 3000)")
    args = ap.parse_args()

    files = sorted(args.data_dir.rglob("*.tif"))
    recs = [r for r in (parse_name(p) for p in files) if r]
    skipped = [p.name for p in files if not parse_name(p)]
    if not recs:
        print(f"No files matching <city>_<year>_<s2|dw>.tif under {args.data_dir}")
        return 1

    print(f"Found {len(recs)} rasters under {args.data_dir}")
    if skipped:
        print(f"Skipped (name pattern): {', '.join(skipped)}")

    results = {}
    print("\nReading...", end="", flush=True)
    for r in recs:
        results[r["path"]] = inspect_s2(r, args.max_dim) if r["kind"] == "s2" \
            else inspect_dw(r, args.max_dim)
        print(".", end="", flush=True)
    print(" done\n")

    # --- 1. pairing -------------------------------------------------------
    print("=" * 74)
    print("1. PAIRING")
    print("=" * 74)
    by_city = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        by_city[r["city"]][r["year"]][r["kind"]] = r
    for city in sorted(by_city):
        years = sorted(by_city[city])
        missing = [f"{y}:{k}" for y in years for k in ("s2", "dw")
                   if k not in by_city[city][y]]
        flag = "OK" if not missing else "MISSING " + ",".join(missing)
        print(f"  {city:<12} years={years}  [{flag}]")

    # --- 2. processing level ---------------------------------------------
    print()
    print("=" * 74)
    print("2. SENTINEL-2 PROCESSING LEVEL (heuristic)")
    print("=" * 74)
    levels = defaultdict(list)
    for r in recs:
        if r["kind"] != "s2":
            continue
        v = results[r["path"]]
        levels[v["level"]].append(f"{r['city']}_{r['year']}")
        print(f"  {r['path'].name:<26} {v['level']:<18} {'; '.join(v['level_why'])}")
        if v["count"] != 5:
            print(f"      !! expected 5 bands, found {v['count']}")
        if v["tags"]:
            print(f"      tags: {v['tags']}")
    if len(levels) > 1:
        print("\n  >> MIXED PROCESSING LEVELS DETECTED. Groups:")
        for k, vs in levels.items():
            print(f"     {k}: {', '.join(vs)}")
        print("     Reflectance distributions will not match. Confirm against your")
        print("     GEE export script before normalizing or running change detection.")

    # --- 3. per-band reflectance ------------------------------------------
    print()
    print("=" * 74)
    print("3. S2 BAND STATS  (p5 / median / p95, valid fraction)")
    print("=" * 74)
    for r in recs:
        if r["kind"] != "s2":
            continue
        v = results[r["path"]]
        print(f"\n  {r['path'].name}   {v['dtype']}  nodata={v['nodata']}")
        for i, st in enumerate(v["bands"]):
            name = S2_BANDS[i] if i < len(S2_BANDS) else f"band{i+1}"
            if st is None:
                print(f"    {name:<12} all-invalid")
                continue
            print(f"    {name:<12} {st['p5']:9.1f} {st['median']:9.1f} "
                  f"{st['p95']:9.1f}   valid={fmt_pct(st['valid_frac'])}")

    # --- 4. alignment ------------------------------------------------------
    print()
    print("=" * 74)
    print("4. ALIGNMENT WITHIN EACH CITY (change detection depends on this)")
    print("=" * 74)
    for city in sorted(by_city):
        sigs = {}
        for year in sorted(by_city[city]):
            for kind, r in by_city[city][year].items():
                v = results[r["path"]]
                sigs[f"{year}_{kind}"] = (v["crs"], v["transform"], v["shape"])
        uniq = {}
        for k, s in sigs.items():
            uniq.setdefault(s, []).append(k)
        if len(uniq) == 1:
            s = next(iter(uniq))
            print(f"  {city:<12} ALIGNED   {s[2][0]}x{s[2][1]}  {s[0]}")
        else:
            print(f"  {city:<12} !! {len(uniq)} DIFFERENT GRIDS")
            for s, members in uniq.items():
                print(f"      {s[2][0]}x{s[2][1]}  {s[0]}  origin={s[1][2]:.1f},{s[1][5]:.1f}")
                print(f"        <- {', '.join(sorted(members))}")
            print("      Reproject/snap to a common grid before change detection.")

    # --- 5. DW class mix / season -----------------------------------------
    print()
    print("=" * 74)
    print("5. DYNAMIC WORLD CLASS MIX  (crops:bare ratio hints at season)")
    print("=" * 74)
    hdr = "  " + f"{'file':<26}" + "".join(f"{n[:6]:>8}" for n in DW_CLASSES.values())
    print(hdr)
    for r in recs:
        if r["kind"] != "dw":
            continue
        v = results[r["path"]]
        row = "".join(f"{100*v['hist'].get(c,0):7.1f}%" for c in DW_CLASSES)
        print(f"  {r['path'].name:<26}{row}")
        if v["unexpected"]:
            print(f"      !! values outside 0-8: {v['unexpected']} "
                  f"(nodata? remap before training)")
        if v["valid_frac"] < 0.98:
            print(f"      valid pixels: {fmt_pct(v['valid_frac'])}")

    print()
    print("  Within one city, a big swing in crops-vs-bare between years usually")
    print("  means the scenes were captured in different parts of the crop calendar,")
    print("  not that the land changed. Check that before reading any transition map.")

    # --- 6. what to do next ------------------------------------------------
    print()
    print("=" * 74)
    print("6. NOTES")
    print("=" * 74)
    print("  - Processing level here is inferred from pixel statistics only. The")
    print("    definitive answer is the collection ID in your GEE export script:")
    print("    COPERNICUS/S2_SR_HARMONIZED = L2A, COPERNICUS/S2_HARMONIZED = L1C.")
    print("  - These GeoTIFFs likely carry no acquisition date. If you composited")
    print("    over a date range in GEE, that range lives in the export script too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
