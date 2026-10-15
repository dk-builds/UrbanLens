"""
Phase 5b: independent validation against hand-labeled points (project
instructions section 8). Three steps:

  sample  draw stratified random points in the held-out cities (2024 date)
  tool    build a self-contained HTML labeling page for those points
  score   score the models AND Dynamic World against the hand labels

Section 8 names Ahmedabad + Coimbatore. Since the 2026-10-06 split change
Coimbatore is a training city and Ahmedabad is out of the corpus, so this uses
the config's current test cities (Hyderabad, Kochi) instead. The intent is the
same: unseen geography only.

Design. Strata are city x class predicted by the 5ch model (the map shown on
the site), with an equal number of points per stratum so small classes
(Bare Soil) get enough points. Every point has a known inclusion probability,
so each sample carries a weight N_h / n_h and every estimate below is
area-weighted. That makes the numbers unbiased for the 8ch model and for Dynamic
World too, not only for the map used to stratify. Point order is shuffled
globally, so labeling only a prefix still gives a valid (smaller) stratified
sample: weights use the number of points actually labeled per stratum.

The labeler must not see any prediction. The labeling page shows only point
locations and Sentinel-2 chips (model input, not output) until a point has been
labeled; only then does it reveal the 5ch and Dynamic World classes there.
Strata live in a separate design file.

Usage:
    python src/independent_validation.py sample
    python src/independent_validation.py tool
    #   open outputs/validation/label_tool.html, label, export the CSV to
    #   data/validation/labels.csv
    python src/independent_validation.py score
"""

import argparse
import base64
import csv
import io
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from dataset import REPO_ROOT, load_config, raster_paths, remap_dw

YEAR = 2024
PRIMARY_TAG = "unet5ch_v2_ablation_best"
MODEL_TAGS = {"5ch": "unet5ch_v2_ablation_best", "8ch": "unet8ch_v2_best"}
POINTS_PER_STRATUM = 30
MIN_SPACING_PX = 10  # 100 m between points in a city

VAL_DATA = REPO_ROOT / "data" / "validation"
DESIGN_CSV = VAL_DATA / "points_design.csv"
LABELS_CSV = VAL_DATA / "labels.csv"
CAPTURE_CACHE = VAL_DATA / "imagery_captures_cache.json"
TOOL_HTML = REPO_ROOT / "outputs" / "validation" / "label_tool.html"
RESULT_JSON = REPO_ROOT / "outputs" / "metrics" / "independent_validation.json"

CHIP_PX = 40      # S2 chip is 40x40 pixels = 400 m
CHIP_SCALE = 5
WAYBACK_CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
WAYBACK_FROM = "2023-01-01"
NODATA = 255


def pred_path(city, tag):
    # The change-detection run for the longest pair holds the 2024 prediction.
    hits = sorted((REPO_ROOT / "outputs" / "change_detection").glob(f"{city}_*_{YEAR}_{tag}_pred_{YEAR}.tif"))
    if not hits:
        raise FileNotFoundError(f"no {YEAR} prediction for {city} / {tag}; run change_detection.py first")
    return hits[0]


def read_full(path, band=1):
    with rasterio.open(path) as r:
        return r.read(band), r.transform, r.crs


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------

def sample_points(config, seed):
    rng = np.random.default_rng(seed)
    classes = config["classes"]["names"]
    ignore = config["classes"]["ignore_index"]
    rows_out = []

    for city in config["cities"]["test"]:
        s2_path, dw_path = raster_paths(city, YEAR)
        pred, transform, crs = read_full(pred_path(city, PRIMARY_TAG))
        dw, _, _ = read_full(dw_path)
        dw = remap_dw(dw, config["classes"]["dw_to_target"], ignore)
        with rasterio.open(s2_path) as r:
            s2_valid = (r.read() != r.nodata).all(axis=0)
        valid = (pred != NODATA) & (dw != ignore) & s2_valid

        taken = []  # (row, col) accepted so far in this city
        for k, name in enumerate(classes):
            cand = np.flatnonzero(valid & (pred == k))
            n_h_pop = cand.size
            if n_h_pop == 0:
                print(f"[{city}] stratum {name}: no pixels, skipped")
                continue
            picks = 0
            for flat in rng.permutation(cand)[:20000]:
                r_, c_ = divmod(int(flat), pred.shape[1])
                if any(abs(r_ - r2) < MIN_SPACING_PX and abs(c_ - c2) < MIN_SPACING_PX for r2, c2 in taken):
                    continue
                taken.append((r_, c_))
                x, y = transform * (c_ + 0.5, r_ + 0.5)
                rows_out.append(dict(city=city, year=YEAR, row=r_, col=c_, x=x, y=y, crs=str(crs),
                                     stratum=name, stratum_pixels=n_h_pop))
                picks += 1
                if picks == POINTS_PER_STRATUM:
                    break
            print(f"[{city}] stratum {name}: {picks} points from {n_h_pop:,} pixels")

    order = rng.permutation(len(rows_out))
    rows_out = [rows_out[i] for i in order]
    for i, row in enumerate(rows_out, 1):
        row["id"] = f"P{i:03d}"
        lon, lat = warp_transform(row["crs"], "EPSG:4326", [row["x"]], [row["y"]])
        row["lon"], row["lat"] = round(lon[0], 7), round(lat[0], 7)
    return rows_out


def cmd_sample(args):
    config = load_config()
    if DESIGN_CSV.exists() and not args.force:
        raise SystemExit(f"{DESIGN_CSV} exists; re-sampling would invalidate any labels. Use --force.")
    rows = sample_points(config, config["seed"])
    VAL_DATA.mkdir(parents=True, exist_ok=True)
    fields = ["id", "city", "year", "lon", "lat", "row", "col", "x", "y", "crs", "stratum", "stratum_pixels"]
    with open(DESIGN_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} points -> {DESIGN_CSV}")


def load_design():
    with open(DESIGN_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("row", "col", "year", "stratum_pixels"):
            r[k] = int(r[k])
        for k in ("x", "y", "lon", "lat"):
            r[k] = float(r[k])
    return rows


# --------------------------------------------------------------------------
# tool
# --------------------------------------------------------------------------

def stretch_limits(src, bands):
    data = src.read(bands, out_shape=(len(bands), src.height // 8, src.width // 8)).astype("float32")
    data[data == src.nodata] = np.nan
    return [np.nanpercentile(b, (2, 98)) for b in data]


def chip_png(src, row, col, bands, limits):
    half = CHIP_PX // 2
    win = Window(col - half, row - half, CHIP_PX, CHIP_PX)
    a = src.read(bands, window=win, boundless=True, fill_value=0).astype("float32")
    out = np.zeros_like(a)
    for i, (lo, hi) in enumerate(limits):
        out[i] = np.clip((a[i] - lo) / (hi - lo), 0, 1)
    img = Image.fromarray((out.transpose(1, 2, 0) * 255).astype(np.uint8))
    img = img.resize((CHIP_PX * CHIP_SCALE,) * 2, Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pixel_square(r, half_px):
    # corners of a (2*half_px+1)-pixel square centred on the point, in lon/lat
    x0, x1 = r["x"] - 5 - 10 * half_px, r["x"] + 5 + 10 * half_px
    y0, y1 = r["y"] - 5 - 10 * half_px, r["y"] + 5 + 10 * half_px
    lons, lats = warp_transform(r["crs"], "EPSG:4326", [x0, x1, x1, x0], [y0, y0, y1, y1])
    return [[round(la, 7), round(lo, 7)] for lo, la in zip(lons, lats)]


def wayback_releases():
    with urllib.request.urlopen(WAYBACK_CONFIG, timeout=30) as resp:
        cfg = json.load(resp)
    rel = []
    for num, v in cfg.items():
        date = v["itemTitle"].split("Wayback ")[-1].rstrip(")")
        if date >= WAYBACK_FROM:
            rel.append(dict(num=int(num), date=date, meta=v["metadataLayerUrl"]))
    return sorted(rel, key=lambda d: d["date"])


def identify_capture(rel, lon, lat):
    d = 0.001
    q = urllib.parse.urlencode(dict(
        geometry=f"{lon},{lat}", geometryType="esriGeometryPoint", sr=4326, layers="all", tolerance=0,
        mapExtent=f"{lon - d},{lat - d},{lon + d},{lat + d}", imageDisplay="400,400,96",
        returnGeometry="false", f="json"))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{rel['meta']}/identify?{q}", timeout=30) as resp:
                res = json.load(resp).get("results", [])
            break
        except OSError:
            if attempt == 2:
                return None
    hits = sorted((r for r in res if r["attributes"].get("SRC_DATE") not in (None, "Null")), key=lambda r: r["layerId"])
    if not hits:
        return None
    a, s = hits[0]["attributes"], str(hits[0]["attributes"]["SRC_DATE"])
    return dict(date=f"{s[:4]}-{s[4:6]}-{s[6:8]}", res=a["SRC_RES"], src=a["NICE_DESC"])


def point_captures(design, releases):
    """For every point, the distinct high-res captures that the Wayback releases
    show there, each with the earliest release that has it. Cached on disk,
    since it is ~14k metadata requests."""
    cache = json.loads(CAPTURE_CACHE.read_text()) if CAPTURE_CACHE.exists() else {}
    todo = [(r, rel) for r in design for rel in releases if f'{r["id"]}|{rel["num"]}' not in cache]
    if todo:
        print(f"querying imagery capture dates: {len(todo)} requests")
        with ThreadPoolExecutor(32) as ex:
            futs = {ex.submit(identify_capture, rel, r["lon"], r["lat"]): f'{r["id"]}|{rel["num"]}' for r, rel in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                cache[futs[fut]] = fut.result()
                if n % 1000 == 0:
                    print(f"  {n}/{len(todo)}")
        CAPTURE_CACHE.write_text(json.dumps(cache))
    out = {}
    for r in design:
        by_date = {}
        for rel in releases:  # sorted oldest first, so the first release seen per capture wins
            c = cache.get(f'{r["id"]}|{rel["num"]}')
            if c and c["date"] not in by_date:
                by_date[c["date"]] = dict(c, release=rel["num"], releaseDate=rel["date"])
        out[r["id"]] = sorted(by_date.values(), key=lambda c: c["date"])
    return out


def cmd_tool(args):
    config = load_config()
    design = load_design()
    points = []
    by_city = {}
    for r in design:
        by_city.setdefault(r["city"], []).append(r)

    classes = config["classes"]["names"]
    for city, rows in by_city.items():
        s2_path, dw_path = raster_paths(city, YEAR)
        # shown only after a point is labeled, so the labels stay blind
        model = read_full(pred_path(city, PRIMARY_TAG))[0]
        dw = remap_dw(read_full(dw_path)[0], config["classes"]["dw_to_target"], config["classes"]["ignore_index"])
        with rasterio.open(s2_path) as src:
            # band order B02 B03 B04 B08 B11 -> true colour 3,2,1; false colour 4,3,2
            lim_tc = stretch_limits(src, [3, 2, 1])
            lim_fc = stretch_limits(src, [4, 3, 2])
            for r in rows:
                points.append(dict(
                    id=r["id"], city=city, lat=r["lat"], lon=r["lon"],
                    px=pixel_square(r, 0), px3=pixel_square(r, 1),
                    tc=chip_png(src, r["row"], r["col"], [3, 2, 1], lim_tc),
                    fc=chip_png(src, r["row"], r["col"], [4, 3, 2], lim_fc),
                    model=classes[model[r["row"], r["col"]]], dw=classes[dw[r["row"], r["col"]]],
                ))
    points.sort(key=lambda p: p["id"])
    releases = wayback_releases()
    caps = point_captures(design, releases)
    for p in points:
        p["caps"] = caps[p["id"]]

    payload = dict(points=points, classes=config["classes"]["names"], releases=releases,
                   target_date=f"{YEAR}-02-15", s2_window=f"{YEAR}-01-01 to {YEAR}-03-31",
                   chip_px=CHIP_PX, chip_scale=CHIP_SCALE)
    template = (Path(__file__).parent / "label_tool_template.html").read_text()
    html = template.replace("/*__PAYLOAD__*/null", json.dumps(payload, separators=(",", ":")))
    TOOL_HTML.parent.mkdir(parents=True, exist_ok=True)
    TOOL_HTML.write_text(html)
    print(f"wrote {TOOL_HTML} ({TOOL_HTML.stat().st_size / 1e6:.1f} MB, {len(points)} points)")


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def load_labels(classes, path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        lab = r["label"].strip()
        if lab in classes:
            out[r["id"]] = dict(label=classes.index(lab), unsure=r.get("unsure", "").strip() in ("1", "true", "True"))
        elif lab:
            out[r["id"]] = dict(label=None, unsure=False)  # "cannot label" -> excluded
    return out


def estimate(ref, maps, strata, stratum_pixels, n_classes):
    """Area-weighted confusion matrix (proportions) for each map, from a
    stratified sample. ref, strata: arrays over samples; maps: {name: array}."""
    w = np.zeros(len(ref))
    for h in np.unique(strata):
        m = strata == h
        w[m] = stratum_pixels[m][0] / m.sum()
    w = w / w.sum()
    out = {}
    for name, pred in maps.items():
        cm = np.zeros((n_classes, n_classes))
        np.add.at(cm, (ref, pred), w)  # rows = reference, cols = map
        out[name] = cm
    return out


def summarize(cm, classes):
    oa = float(np.trace(cm))
    per = {}
    ious = []
    for k, name in enumerate(classes):
        tp, ref_k, map_k = cm[k, k], cm[k].sum(), cm[:, k].sum()
        iou = tp / (ref_k + map_k - tp) if ref_k + map_k > 0 else None
        per[name] = dict(
            users_accuracy=tp / map_k if map_k > 0 else None,      # precision
            producers_accuracy=tp / ref_k if ref_k > 0 else None,  # recall
            iou=iou, ref_area_frac=float(ref_k), map_area_frac=float(map_k))
        if iou is not None:
            ious.append(iou)
    return dict(overall_accuracy=oa, miou=float(np.mean(ious)), per_class=per)


def bootstrap(ref, maps, strata, stratum_pixels, n_classes, reps, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == h) for h in np.unique(strata)]
    draws = {name: dict(oa=[], miou=[]) for name in maps}
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        cms = estimate(ref[idx], {k: v[idx] for k, v in maps.items()}, strata[idx], stratum_pixels[idx], n_classes)
        for name, cm in cms.items():
            s = summarize(cm, range(n_classes))
            draws[name]["oa"].append(s["overall_accuracy"])
            draws[name]["miou"].append(s["miou"])
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in draws.items()}


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def score_subset(sel, design, labels, values, classes, reps, seed):
    ids = [r["id"] for r in sel]
    ref = np.array([labels[i]["label"] for i in ids])
    strata = np.array([f'{r["city"]}/{r["stratum"]}' for r in sel])
    stratum_pixels = np.array([r["stratum_pixels"] for r in sel], dtype=float)
    maps = {name: np.array([values[name][i] for i in ids]) for name in values}
    cms = estimate(ref, maps, strata, stratum_pixels, len(classes))
    boot = bootstrap(ref, maps, strata, stratum_pixels, len(classes), reps, seed)

    res = dict(n_points=len(ids), maps={})
    for name, cm in cms.items():
        s = summarize(cm, classes)
        s["overall_accuracy_ci95"] = ci(boot[name]["oa"])
        s["miou_ci95"] = ci(boot[name]["miou"])
        s["confusion_matrix_area_frac"] = np.round(cm, 5).tolist()
        s["n_correct_unweighted"] = int((maps[name] == ref).sum())
        res["maps"][name] = s
    for name in values:
        if name == "dynamic_world":
            continue
        d = boot[name]["oa"] - boot["dynamic_world"]["oa"]
        res[f"{name}_minus_dw_oa"] = dict(
            estimate=round(res["maps"][name]["overall_accuracy"] - res["maps"]["dynamic_world"]["overall_accuracy"], 4),
            ci95=ci(d), frac_boot_above_zero=round(float((d > 0).mean()), 3))
    res["ref_counts_unweighted"] = {c: int((ref == k).sum()) for k, c in enumerate(classes)}
    return res


def cmd_score(args):
    config = load_config()
    classes = config["classes"]["names"]
    design = load_design()
    labels = load_labels(classes, args.labels)

    values = {"dynamic_world": {}}
    values.update({name: {} for name in MODEL_TAGS})
    for city in sorted({r["city"] for r in design}):
        rows = [r for r in design if r["city"] == city]
        _, dw_path = raster_paths(city, YEAR)
        dw = remap_dw(read_full(dw_path)[0], config["classes"]["dw_to_target"], config["classes"]["ignore_index"])
        sources = {"dynamic_world": dw}
        sources.update({name: read_full(pred_path(city, tag))[0] for name, tag in MODEL_TAGS.items()})
        for name, arr in sources.items():
            for r in rows:
                values[name][r["id"]] = int(arr[r["row"], r["col"]])

    labeled = [r for r in design if r["id"] in labels and labels[r["id"]]["label"] is not None]
    excluded = [r["id"] for r in design if r["id"] in labels and labels[r["id"]]["label"] is None]
    unlabeled = len(design) - len(labeled) - len(excluded)
    if not labeled:
        raise SystemExit("no usable labels yet")
    small = {}
    for r in labeled:
        small[(r["city"], r["stratum"])] = small.get((r["city"], r["stratum"]), 0) + 1
    thin = {f"{c}/{s}": n for (c, s), n in small.items() if n < 5}

    results = dict(
        note=("Area-weighted estimates from a stratified random sample (strata = city x 5ch-model class, "
              f"{YEAR} composite). Reference = hand labels on high-resolution imagery. Rows of the "
              "confusion matrices are reference, columns are map; cells are estimated area fractions."),
        n_design=len(design), n_labeled=len(labeled), n_cannot_label=len(excluded), n_unlabeled=unlabeled,
        thin_strata=thin,
        pooled=score_subset(labeled, design, labels, values, classes, args.reps, config["seed"]),
        sure_only=score_subset([r for r in labeled if not labels[r["id"]]["unsure"]],
                               design, labels, values, classes, args.reps, config["seed"]),
    )
    for city in sorted({r["city"] for r in labeled}):
        results[city] = score_subset([r for r in labeled if r["city"] == city], design, labels, values,
                                     classes, args.reps, config["seed"])

    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"labeled {len(labeled)} / {len(design)} (cannot label {len(excluded)}, not yet {unlabeled})")
    if thin:
        print(f"WARNING strata with <5 labeled points (weights unstable): {thin}")
    for scope in ["pooled", "sure_only"] + sorted({r["city"] for r in labeled}):
        res = results[scope]
        print(f"\n== {scope} (n={res['n_points']}) ==")
        print(f"{'map':<14}{'OA':>7}  {'95% CI':<17}{'mIoU':>6}  " + "  ".join(f"{c[:8]:>8}" for c in classes))
        for name, s in res["maps"].items():
            ious = "  ".join(f"{s['per_class'][c]['iou']:>8.3f}" if s["per_class"][c]["iou"] is not None
                             else f"{'-':>8}" for c in classes)
            print(f"{name:<14}{s['overall_accuracy']:>7.3f}  {str(s['overall_accuracy_ci95']):<17}"
                  f"{s['miou']:>6.3f}  {ious}")
        for name in MODEL_TAGS:
            d = res[f"{name}_minus_dw_oa"]
            print(f"{name} - DW overall accuracy: {d['estimate']:+.3f}  95% CI {d['ci95']}")
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--force", action="store_true")
    sub.add_parser("tool")
    p = sub.add_parser("score")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--labels", type=Path, default=LABELS_CSV)
    p.add_argument("--out", type=Path, default=RESULT_JSON)
    args = ap.parse_args()
    {"sample": cmd_sample, "tool": cmd_tool, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
