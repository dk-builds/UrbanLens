# UrbanLens: urban sprawl detection for Indian cities

A U-Net that labels every 10 m Sentinel-2 pixel as Urban, Vegetation, Water,
Agriculture or Bare Soil. It is trained on five Indian cities and tested on two
cities it never saw (Hyderabad, Kochi), then used to measure how much built-up
land those cities added between 2019 and 2024. Results are checked against 300
points labeled by hand on high-resolution imagery, not only against the labels
the model was trained on.

**Status: work in progress.** Data pipeline, model, evaluation, independent
validation, change detection, interactive map and API are done. Next steps are
listed under [Roadmap](#roadmap).

| | |
|---|---|
| Held-out cities, agreement with Dynamic World | mIoU 0.73 (5-band model) |
| Hand-labeled points, overall accuracy | **86.7%**, vs 80.0% for Dynamic World itself |
| Hyderabad built-up area, 2019 to 2024 | **+128 km²** net (+14.8%) |
| Inference | ~85 s on CPU for a 44 x 44 km city (Docker, 1.4 GB peak memory) |

## How it works

```
Google Earth Engine          PyTorch                         Outputs
-------------------          -------                         -------
Sentinel-2 L2A, 5 bands  ->  U-Net, 256 px patches,      ->  class map per date
(Jan-Mar median, 10 m)       trained on 5 cities              change map + km² stats
Dynamic World labels     ->  (labels remapped to 5 classes)   interactive map, REST API
(same window, mode)
```

- **Imagery:** Sentinel-2 L2A surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`),
  bands B02, B03, B04, B08, B11, median composite of Jan to Mar (dry season,
  little cloud). Export script: [scripts/export_gee.py](scripts/export_gee.py).
- **Labels:** [Dynamic World](https://dynamicworld.app) (Google/WRI), mode
  composite over the same window, 9 classes mapped to 5.
- **Split by city, not by pixel:** train on Bengaluru, Chennai, Pune,
  Coimbatore and Vizag; test on Hyderabad and Kochi, which contribute no
  training patches. Validation uses a held-out eastern strip of each training
  city. Band statistics come from training cities only and are frozen in
  [configs/config.yaml](configs/config.yaml).
- **Model:** 4-level U-Net trained from scratch, class-weighted cross-entropy,
  AdamW with cosine schedule, early stopping on validation mIoU, MLflow
  tracking. 6,788 training and 1,518 validation patches.
- **Ablation:** the same model with three spectral indices added (NDVI, NDBI,
  MNDWI; 8 input channels) against the 5 raw bands.

## Results

### 1. Unseen cities, scored against Dynamic World

Per-pixel scores over the full Hyderabad (2019, 2021, 2024) and Kochi (2019,
2024) rasters. These measure agreement with Dynamic World, the labels the model
learned from, not correctness.

| IoU | Urban | Vegetation | Water | Agriculture | Bare Soil | **mIoU** |
|---|---|---|---|---|---|---|
| 5 bands | 0.899 | 0.740 | 0.972 | 0.629 | 0.405 | **0.729** |
| 5 bands + 3 indices | 0.896 | 0.728 | 0.968 | 0.583 | 0.379 | 0.711 |

**The spectral indices did not help.** The 5-band model is ahead on every
held-out city and year, most clearly on Agriculture and Bare Soil. One training
run per model, so a gap this size may partly be seed noise.

### 2. Independent check against hand labels

Dynamic World agrees with expert labels about 74% of the time, so matching it
closely says little about real accuracy. To measure accuracy, 300 points were
labeled by eye on sub-metre Esri World Imagery, using the capture date closest
to the Jan to Mar 2024 composite (Esri Wayback archive), without seeing any
prediction. Points are a stratified random sample (30 per city and predicted
class) and every estimate is area-weighted, so the numbers stand for the whole
of each city. Code: [src/independent_validation.py](src/independent_validation.py).

| Overall accuracy (95% CI) | Both cities | Hyderabad | Kochi |
|---|---|---|---|
| Dynamic World | 80.0% (75 to 85) | 70.5% | 89.5% |
| **5-band model** | **86.7% (81 to 92)** | **80.9%** | **92.6%** |
| 8-channel model | 84.8% (79 to 90) | 78.8% | 90.7% |
| 5-band minus Dynamic World | **+6.7 pts (+3.4 to +10.0)** | +10.4 (+5.7 to +15.7) | +3.1 (-0.7 to +6.5) |

**The model is more accurate than its own training labels**, clearly in
Hyderabad and overall, not significantly in Kochi. Point by point, it is right
where Dynamic World is wrong on 56 points and the reverse on 17.

### 3. Change detection, Hyderabad 2019 to 2024

| | Built-up 2019 | Built-up 2024 | Net change |
|---|---|---|---|
| 5-band model | 867 km² | 995 km² | **+128 km²** |
| Dynamic World | 946 km² | 1,074 km² | +128 km² |

New built-up land came mostly from vegetation and scrub (54%) and farmland
(36%). Kochi 2019 to 2024: +48 km² (Dynamic World +47 km²).

## Limitations

- **Urban is over-mapped and Bare Soil under-mapped.** Against the hand labels
  the model maps 38% of the area as Urban where the reference is 31%, and finds
  only 18% of real bare ground. Bright bare ground (cleared plots, rock,
  plotted real-estate layouts) looks like rooftops and roads in these 5 bands.
- **Year-to-year change carries the same error.** Hyderabad shows 37 km² of
  built-up "loss" from 2019 to 2024. Checking the largest patches on archived
  high-resolution imagery shows dry, bare ground in 2019 that was read as
  built-up and had greened by 2024, not demolition. Dynamic World makes the
  same mistake, so agreeing with it does not confirm the change. Net change is
  the more reliable figure; the shorter intervals (2019 to 2021, 2021 to 2024)
  disagree with Dynamic World.
- **Change itself is not hand-validated;** the 300 points check the 2024 map.
- **One labeler, one training run per model, two test cities.**
- **Tile seams:** inference uses non-overlapping 256 px tiles, so a crop can
  differ from the same area cut from a full-city run (about 2.7% of pixels in
  one test, mostly near tile edges).

## API

FastAPI service: upload a 5-band GeoTIFF (B02, B03, B04, B08, B11, reflectance
0 to 10000, 10 m), get the class map back as a GeoTIFF on the same grid, or a
JSON summary of class areas.

```bash
# needs the checkpoint at outputs/checkpoints/unet5ch_v2_ablation_best.pt
docker compose up --build            # URBANLENS_PORT=8010 if 8000 is taken
curl localhost:8000/health
curl -F file=@examples/kochi_2024_sample.tif localhost:8000/classify -o classes.tif
curl -F file=@examples/kochi_2024_sample.tif localhost:8000/classify/summary
```

| Endpoint | |
|---|---|
| `GET /health` | liveness and whether the model loaded |
| `GET /model` | classes, expected bands and limits |
| `POST /classify` | class GeoTIFF (uint8, 255 = no data, colour table included) |
| `POST /classify/summary` | pixels, share and km² per class |

Wrong band count, unreadable files and rasters over 25M pixels are rejected;
a non-10 m pixel size or 0 to 1 scaled reflectance returns a warning. The
image is CPU-only PyTorch, runs as a non-root user and mounts the checkpoint
read-only. Interactive docs at `/docs`.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/export_gee.py --project <gcp-project-id>      # rasters -> data/raw/
python src/dataset.py --compute-stats                          # band stats -> config
python src/dataset.py --build-index                            # patch sample
python src/train.py --in-channels 5                            # 8 = bands + indices
python src/evaluate.py --checkpoint outputs/checkpoints/unet5ch_v2_ablation_best.pt
python src/change_detection.py --checkpoint outputs/checkpoints/unet5ch_v2_ablation_best.pt \
    --city hyderabad --year-a 2019 --year-b 2024
python src/make_map.py --city hyderabad --year-a 2019 --year-b 2024
python src/independent_validation.py score                     # uses data/validation/labels.csv
```

Raw rasters (2.8 GB) and checkpoints are not in the repository.

## Repository

```
configs/config.yaml          frozen settings: band stats, class mapping, city split
scripts/export_gee.py        Earth Engine export of Sentinel-2 + Dynamic World
src/dataset.py               patch sampling, normalization, splits
src/spectral_indices.py      NDVI / NDBI / MNDWI
src/model.py                 U-Net
src/train.py                 training with MLflow
src/predict.py               tiled inference on any 5-band GeoTIFF
src/evaluate.py              held-out city metrics vs Dynamic World
src/independent_validation.py  hand-label sampling, labeling page, scoring
src/change_detection.py      transition matrices, expansion maps, km² stats
src/make_map.py              interactive Leaflet map
src/api.py                   FastAPI service
data/validation/             sample design and the 300 hand labels
outputs/metrics/             all reported numbers (JSON)
outputs/maps/                interactive maps (open the HTML)
examples/                    5 x 5 km Kochi sample for the API
```

## Roadmap

- Separate bare ground from built-up better: add the SWIR-2 band (B12) and
  train on Dynamic World's class probabilities instead of its single label.
- Season-matched composites, so a dry year is not compared with a wet one.
- Sentinel-2-pretrained encoder (SSL4EO-S12 ResNet-50).
- Overlapping tiles to remove seams; repeat seeds for the 5 vs 8 channel result.
- A second labeler to measure inter-rater agreement.

## Data and credits

Contains modified Copernicus Sentinel data (2017 to 2025). Dynamic World V1,
Google and World Resources Institute, CC BY 4.0. Reference imagery for labeling:
Esri World Imagery and World Imagery Wayback (Maxar, Vantor). Basemaps: Esri,
OpenStreetMap contributors.
