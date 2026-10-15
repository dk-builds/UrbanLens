#!/usr/bin/env python3
"""
GEE export script for the UrbanLens corpus: 5-band Sentinel-2 L2A + matching
Dynamic World labels, one GeoTIFF pair per city/year, written to data/raw/.

Reconstructed after the fact (2026-08-12) — the script originally used to
pull Bengaluru/Hyderabad/Chennai/Pune/Ahmedabad/Coimbatore is lost, so the
AOIs for those 6 below are backed out of the *existing* rasters' own
georeferencing (exact bounds + CRS read via rasterio), not re-guessed. Only
the date-range convention (Jan 1 - Mar 31 dry season, minimal cloud/monsoon
interference) is an assumption going forward — there's no way to recover
what date range the original 6 actually used. If a from-scratch re-export of
the whole corpus is ever needed, everything required to reproduce it is
here.

Sentinel-2: COPERNICUS/S2_SR_HARMONIZED (L2A, surface reflectance, 0-10000
int16 scale), bands B2/B3/B4/B8/B11, SCL-based cloud/shadow/snow masking,
median composite of scenes with < 40% cloudy-pixel percentage.

Dynamic World: GOOGLE/DYNAMICWORLD/V1 'label' band, mode composite over the
same date range as the matched S2 composite (date-matched labels, not a
static annual snapshot — see project instructions section 3).

Usage:
    python scripts/export_gee.py --project <gcp-project-id> --city coimbatore --year 2019
    python scripts/export_gee.py --project <gcp-project-id> --city coimbatore
    python scripts/export_gee.py --project <gcp-project-id> --city all   # re-export everything
"""

import argparse
import io
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import ee
import rasterio
import requests
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_bounds

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# getDownloadURL rejects requests over 48 MiB uncompressed. Cities here are
# ~150-300MB uncompressed (5 bands x ~4400x4500 int16), so downloads are
# tiled into a grid and mosaicked locally with rasterio.
MAX_TILE_BYTES = 35 * 1024 * 1024

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
S2_BANDS = ["B2", "B3", "B4", "B8", "B11"]
CLOUD_PCT_MAX = 40
DW_NODATA = 255  # outside DW's valid 0-8 class range

# AOIs as [west, south, east, north] in WGS84, except entries carrying
# `utm_bounds`, which are used directly (see export_city_year). Kochi and Vizag
# are ~44km squares centered on the city in their native UTM zone, with
# utm_bounds set so both years share one pixel grid.
CITY_AOIS = {
    "bengaluru": dict(utm_bounds=(759060.0, 1412890.0, 802940.0, 1457620.0),
                      crs="EPSG:32643", years=[2017, 2019, 2021, 2024, 2025]),
    "chennai": dict(utm_bounds=(399090.0, 1423920.0, 442600.0, 1468290.0),
                    crs="EPSG:32644", years=[2019, 2021, 2024]),
    # coimbatore: added 2019 (2026-09-26) — now the Phase 7 change-detection
    # demo city, taking over Mumbai's former role (see project state notes).
    # Mumbai's own AOI/years entry removed along with the rest of its data.
    # utm_bounds pinned to the exact grid of the pre-existing coimbatore_2024
    # raster (read via rasterio) rather than relying on `bounds` (WGS84) —
    # see export_city_year's utm_bounds comment for why the round-trip
    # doesn't reproduce this exactly.
    "coimbatore": dict(bounds=[76.7576, 10.8174, 77.1630, 11.2227],
                        utm_bounds=(692150.0, 1196630.0, 736170.0, 1241180.0),
                        crs="EPSG:32643", years=[2019, 2024]),
    "hyderabad": dict(utm_bounds=(209590.0, 1900950.0, 252710.0, 1945810.0),
                      crs="EPSG:32644", years=[2019, 2021, 2024]),
    "kochi": dict(utm_bounds=(616930.0, 1076070.0, 660930.0, 1120070.0),
                  crs="EPSG:32643", years=[2019, 2024]),
    "pune": dict(utm_bounds=(358390.0, 2025840.0, 400900.0, 2070380.0),
                 crs="EPSG:32643", years=[2019, 2021, 2024]),
    "vizag": dict(utm_bounds=(721800.0, 1935020.0, 765800.0, 1979020.0),
                  crs="EPSG:32644", years=[2019, 2024]),
}


def mask_s2_clouds(img):
    scl = img.select("SCL")
    # 3 cloud shadow, 8/9 cloud medium/high prob, 10 cirrus, 11 snow/ice
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return img.updateMask(bad.Not())


def build_s2_composite(aoi, year):
    start, end = f"{year}-01-01", f"{year}-03-31"
    coll = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT_MAX))
        .map(mask_s2_clouds)
    )
    n = coll.size().getInfo()
    if n == 0:
        raise RuntimeError(f"No S2 scenes found for {start}..{end} under {CLOUD_PCT_MAX}% cloud")
    composite = coll.select(S2_BANDS).median().clip(aoi)
    return composite.toInt16(), n, (start, end)


def build_dw_composite(aoi, date_range):
    start, end = date_range
    coll = ee.ImageCollection(DW_COLLECTION).filterBounds(aoi).filterDate(start, end)
    n = coll.size().getInfo()
    if n == 0:
        raise RuntimeError(f"No Dynamic World scenes found for {start}..{end}")
    # mode() leaves pixels with no contributing scene masked. unmask to a
    # sentinel outside the valid 0-8 class range *before* export — DW class 0
    # is Water, so if we let GEE/GDAL pick a default nodata value (commonly
    # 0 for byte rasters), every real Water pixel becomes indistinguishable
    # from "no data" once written to disk.
    composite = coll.select("label").mode().clip(aoi).unmask(DW_NODATA)
    return composite.toByte(), n


def fetch_tile(img, region, crs, scale):
    url = img.getDownloadURL(
        {"region": region, "crs": crs, "scale": scale, "format": "GEO_TIFF"}
    )
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    content = r.content
    if content[:2] == b"PK":  # zip wrapper for some responses
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = z.namelist()
            if len(names) != 1:
                raise RuntimeError(f"Expected 1 file in zip, got {names}")
            content = z.read(names[0])
    return content


def download_image(img, utm_bounds, crs, out_path, bands, dtype_size, scale=10, nodata=None):
    """Download img (already clipped to utm_bounds), tiling the request grid
    as needed to stay under GEE's per-request size cap, then mosaic.

    utm_bounds must be the *same* projected rectangle the image was clipped
    to when it was built (see export_city_year) — computing this bound
    separately from the clip geometry (e.g. re-deriving it from a WGS84 box)
    lets the two disagree by a sliver at the projection's corners, which
    shows up as a strip of nodata along one edge of the mosaic.
    """
    minx, miny, maxx, maxy = utm_bounds
    width_px = (maxx - minx) / scale
    height_px = (maxy - miny) / scale
    total_bytes = width_px * height_px * bands * dtype_size
    n = max(1, math.ceil(math.sqrt(total_bytes / MAX_TILE_BYTES)))

    step_x = (maxx - minx) / n
    step_y = (maxy - miny) / n

    with tempfile.TemporaryDirectory() as tmp:
        tile_paths = []
        for i in range(n):
            for j in range(n):
                tminx = minx + i * step_x
                tmaxx = minx + (i + 1) * step_x
                tminy = miny + j * step_y
                tmaxy = miny + (j + 1) * step_y
                region = ee.Geometry.Rectangle(
                    [tminx, tminy, tmaxx, tmaxy], proj=crs, geodesic=False
                )
                content = fetch_tile(img, region, crs, scale)
                tp = Path(tmp) / f"tile_{i}_{j}.tif"
                tp.write_bytes(content)
                tile_paths.append(tp)

        if n == 1:
            srcs = [rasterio.open(tile_paths[0])]
            mosaic = srcs[0].read()
            out_transform = srcs[0].transform
        else:
            srcs = [rasterio.open(p) for p in tile_paths]
            mosaic, out_transform = rio_merge(srcs)

        profile = srcs[0].profile
        profile.update(
            height=mosaic.shape[1], width=mosaic.shape[2], transform=out_transform
        )
        if nodata is not None:
            profile["nodata"] = nodata
        for s in srcs:
            s.close()
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)


def export_city_year(city, year, cfg, bands="both"):
    crs = cfg["crs"]
    if "utm_bounds" in cfg:
        # Bypasses the WGS84 round-trip below entirely. transform_bounds
        # computes the *bounding box* of a reprojected rectangle, which is
        # not a perfect inverse of the original rectangle (meridian
        # convergence bulges the WGS84 box outward) — round-tripping a
        # city's stored WGS84 bounds back to UTM lands 260-290m off from
        # its own previously-exported raster, discovered exporting
        # coimbatore_2019 to pair with the pre-existing coimbatore_2024
        # (2026-09-26). Harmless for a standalone export, but it breaks the
        # exact pixel-grid alignment change detection needs between two
        # years of the same city. Any city where that alignment matters
        # should carry native-CRS `utm_bounds` (read directly off the
        # reference raster) instead of relying on `bounds` (WGS84).
        utm_bounds = cfg["utm_bounds"]
    else:
        utm_bounds = transform_bounds("EPSG:4326", crs, *cfg["bounds"])
    # geodesic=False: straight edges in the *projected* CRS, matching the
    # tile grid used at download time exactly (see download_image docstring).
    aoi = ee.Geometry.Rectangle(list(utm_bounds), proj=crs, geodesic=False)

    print(f"[{city} {year}] building S2 composite...")
    s2_img, s2_n, date_range = build_s2_composite(aoi, year)
    print(f"[{city} {year}] {s2_n} S2 scenes in {date_range[0]}..{date_range[1]}")

    print(f"[{city} {year}] building Dynamic World composite (date-matched)...")
    dw_img, dw_n = build_dw_composite(aoi, date_range)
    print(f"[{city} {year}] {dw_n} DW scenes")

    s2_path = OUT_DIR / f"{city}_{year}_s2.tif"
    dw_path = OUT_DIR / f"{city}_{year}_dw.tif"

    if bands in ("s2", "both"):
        print(f"[{city} {year}] downloading S2 -> {s2_path}")
        download_image(s2_img, utm_bounds, crs, s2_path, bands=len(S2_BANDS), dtype_size=2)
    if bands in ("dw", "both"):
        print(f"[{city} {year}] downloading DW -> {dw_path}")
        download_image(dw_img, utm_bounds, crs, dw_path, bands=1, dtype_size=1, nodata=DW_NODATA)
    sizes = []
    if s2_path.exists():
        sizes.append(f"{s2_path.stat().st_size/1e6:.1f} MB s2")
    if dw_path.exists():
        sizes.append(f"{dw_path.stat().st_size/1e6:.1f} MB dw")
    print(f"[{city} {year}] done ({', '.join(sizes)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="GCP project ID for ee.Initialize")
    ap.add_argument("--city", required=True, choices=list(CITY_AOIS) + ["all"])
    ap.add_argument("--year", type=int, default=None, help="single year; default = all configured years for the city")
    ap.add_argument("--bands", choices=["s2", "dw", "both"], default="both")
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cities = list(CITY_AOIS) if args.city == "all" else [args.city]
    for city in cities:
        cfg = CITY_AOIS[city]
        years = [args.year] if args.year else cfg["years"]
        for year in years:
            t0 = time.time()
            try:
                export_city_year(city, year, cfg, bands=args.bands)
            except Exception as e:
                print(f"[{city} {year}] FAILED: {e}", file=sys.stderr)
                continue
            print(f"[{city} {year}] took {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    sys.exit(main())
