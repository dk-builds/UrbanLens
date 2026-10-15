"""
NDVI / NDBI / MNDWI, computed from the 5-band S2 stack (B02, B03, B04, B08, B11).

Single code path for both training (dataset.py) and inference (api.py,
predict.py) — see project instructions section 9 for why these are input
features rather than post-hoc correction rules.

Indices are computed on *normalized* reflectance (post band mean/std scaling
is NOT what's meant here — "after band normalization" in the instructions
means after the raw DN -> reflectance scaling GEE already applies, i.e. on
the 0-1-ish surface reflectance values, not on z-scored patches). Called
before the dataset's z-score normalization step, since the indices' own
[-1, 1] bounds would otherwise be destroyed by subtracting a band mean.
"""

import numpy as np

BAND_ORDER = ["B02_blue", "B03_green", "B04_red", "B08_nir", "B11_swir"]
EPS = 1e-6


def compute_indices(bands):
    """bands: array (5, H, W) or (H, W, 5) in surface-reflectance units
    (S2_SR_HARMONIZED, ~0-1 after /10000 scaling, or raw DN — any consistent
    scale works since every term in each ratio shares it).

    Returns (3, H, W) float32: NDVI, NDBI, MNDWI, each in [-1, 1].
    """
    channel_first = bands.shape[0] == 5
    if channel_first:
        blue, green, red, nir, swir = bands
    else:
        blue, green, red, nir, swir = np.moveaxis(bands, -1, 0)

    blue, green, red, nir, swir = (
        a.astype("float32") for a in (blue, green, red, nir, swir)
    )

    ndvi = (nir - red) / (nir + red + EPS)
    ndbi = (swir - nir) / (swir + nir + EPS)
    mndwi = (green - swir) / (green + swir + EPS)

    stacked = np.stack([ndvi, ndbi, mndwi], axis=0 if channel_first else -1)
    return np.clip(stacked, -1.0, 1.0)
