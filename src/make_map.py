"""
Phase 8, interactive Folium map of one city's change-detection result.

Writes a single self-contained HTML file (layer images are embedded as data
URIs), themed to match the portfolio site, so it can be dropped into an
<iframe> with no other hosting. Layers: new and unchanged built-up, and the predicted
land cover at each date, over switchable basemaps.

A temporary colour test tool (picker + opacity slider per colour, log/copy as JSON)
is included; it shows on standalone pages and, when embedded, only with "colors" in
the iframe hash or the host page URL, or after Shift+C inside the map.
--no-color-tool omits it.

Embed mode (active when the page is inside an iframe, or the URL hash contains
"embed", e.g. "#embed&pad=T,R,B,L&ui=T,R,B,L" in CSS px): re-fits the overlay into the
padded box with no animation, posts a 'ready' message to the parent with the
overlay's pixel rect, posts a 'view' message on every moveend, answers a 'reset'
message, and forwards Escape. "pad" insets only the fit box; "ui=T,R,B,L" insets the
chrome (defaults to pad). On desktop widths the panels sit top-left at (24, T+24) and
the zoom control moves to the top-right beside the layers toggle.

--particles also writes a coarse class grid for both dates (same EPSG:3857
warp and bounds as the overlays) for the site's point-cloud morph.

Usage:
    python src/make_map.py --city hyderabad --year-a 2019 --year-b 2024 --particles
"""

import argparse
import base64
import io
import json
import os

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

import folium
from folium.raster_layers import ImageOverlay

from dataset import REPO_ROOT, load_config

CHANGE_DIR = REPO_ROOT / "outputs" / "change_detection"
MAP_DIR = REPO_ROOT / "outputs" / "maps"
VALIDATION_JSON = REPO_ROOT / "outputs" / "metrics" / "independent_validation.json"
NODATA = 255
DST_CRS = "EPSG:3857"
MAX_PARTICLES = 32768


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# Colours and opacities chosen in the on-page colour tool (user's log, 2026-10-09).
CLASS_COLORS = {
    "Urban": hex_rgb("#e84a5f"),
    "Vegetation": hex_rgb("#174a2d"),
    "Water": hex_rgb("#6b8abd"),
    "Agriculture": hex_rgb("#57bc88"),
    "Bare Soil": hex_rgb("#b07cc6"),
}
GAIN_COLOR = hex_rgb("#ea4d5f")
STABLE_URBAN_COLOR = hex_rgb("#6e2a35")
DEFAULT_OPACITY = {  # percent
    "new_built_up": 78, "unchanged_built_up": 59,
    "urban": 66, "vegetation": 65, "water": 69, "agriculture": 65, "bare_soil": 69,
}


def rgb_hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


def make_color_specs(class_names):
    """Every colour editable in the on-page test tool. `code` is the value stored
    in that layer's index image; `opacity` is a percentage."""
    specs = [
        dict(key="new_built_up", label="New built-up", legend="New built-up", group="change", code=1,
             hex=rgb_hex(GAIN_COLOR), opacity=DEFAULT_OPACITY["new_built_up"]),
        dict(key="unchanged_built_up", label="Unchanged built-up", legend="Unchanged built-up",
             group="change", code=2, hex=rgb_hex(STABLE_URBAN_COLOR),
             opacity=DEFAULT_OPACITY["unchanged_built_up"]),
    ]
    for i, name in enumerate(class_names):
        key = name.lower().replace(" ", "_")
        specs.append(dict(key=key, label=name, legend=name, group="lc", code=i,
                          hex=rgb_hex(CLASS_COLORS[name]), opacity=DEFAULT_OPACITY[key]))
    return specs


ESRI_DARK_GRAY = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                  "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}")
ESRI_IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
                "MapServer/tile/{z}/{y}/{x}")
FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900'
             '&display=swap" rel="stylesheet">')


def to_web_mercator(path, max_dim, resampling):
    """Read band 1, warp to EPSG:3857 at <= max_dim pixels. Returns (array,
    lat/lon bounds [[south, west], [north, east]])."""
    with rasterio.open(path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds)
        scale = max(w, h) / max_dim
        if scale > 1:
            w, h = int(w / scale), int(h / scale)
            transform, w, h = calculate_default_transform(
                src.crs, DST_CRS, src.width, src.height, *src.bounds,
                dst_width=w, dst_height=h)
        dst = np.full((h, w), NODATA, dtype=np.uint8)
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=DST_CRS,
                  resampling=resampling, src_nodata=NODATA, dst_nodata=NODATA)
    west, south, east, north = transform_bounds(DST_CRS, "EPSG:4326", *array_bounds(h, w, transform))
    return dst, [[south, west], [north, east]]


def png_data_uri(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def change_codes(pred_a, pred_b):
    """Per-pixel code for the change layer: 0 nothing drawn, 1 new built-up,
    2 unchanged built-up, 255 invalid. Built-up loss is deliberately not drawn
    (it is mostly per-date classification noise), so those pixels stay 0."""
    codes = np.zeros(pred_a.shape, dtype=np.uint8)
    valid = (pred_a != NODATA) & (pred_b != NODATA)
    codes[valid & (pred_a == 0) & (pred_b == 0)] = 2
    codes[valid & (pred_a != 0) & (pred_b == 0)] = 1
    codes[~valid] = NODATA
    return codes


def rgba_from_codes(codes, specs, group):
    table = np.zeros((256, 4), dtype=np.uint8)
    for sp in specs:
        if sp["group"] == group:
            table[sp["code"]] = (*hex_rgb(sp["hex"]), round(255 * sp["opacity"] / 100))
    return table[codes]


def index_png_uri(codes):
    """Lossless image carrying the code in R (alpha 255 = valid), so the page can
    recolour a layer in the browser without a re-export."""
    rgba = np.zeros((*codes.shape, 4), dtype=np.uint8)
    valid = codes != NODATA
    rgba[..., 0] = np.where(valid, codes, 0)
    rgba[..., 3] = np.where(valid, 255, 0)
    return png_data_uri(rgba)


def block_mode(arr, gw, gh, num_classes):
    """Most common class per output cell (nodata ignored; all-nodata -> 255)."""
    h, w = arr.shape
    ys = np.linspace(0, h, gh + 1).astype(int)
    xs = np.linspace(0, w, gw + 1).astype(int)
    out = np.full((gh, gw), NODATA, dtype=np.uint8)
    for i in range(gh):
        for j in range(gw):
            blk = arr[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel()
            blk = blk[blk != NODATA]
            if blk.size:
                out[i, j] = np.bincount(blk, minlength=num_classes).argmax()
    return out


def write_particles(pred_a, pred_b, bounds, class_names, city, year_a, year_b, tag):
    h, w = pred_a.shape
    scale = (MAX_PARTICLES / (w * h)) ** 0.5
    gw, gh = max(1, int(w * scale)), max(1, int(h * scale))
    while gw * gh > MAX_PARTICLES:
        gw -= 1
    a = block_mode(pred_a, gw, gh, len(class_names))
    b = block_mode(pred_b, gw, gh, len(class_names))
    stem = MAP_DIR / f"{city}_{year_a}_{year_b}_particles"
    np.concatenate([a.ravel(), b.ravel()]).astype(np.uint8).tofile(f"{stem}.bin")
    meta = {"city": city, "years": [year_a, year_b], "width": gw, "height": gh,
            "classes": class_names, "bounds": bounds, "tag": tag, "nodata": NODATA}
    (stem.parent / f"{stem.name}.json").write_text(json.dumps(meta, indent=2))
    return gw, gh, str(stem)


def swatch(spec):
    r, g, b = hex_rgb(spec["hex"])
    return (f'<div class="row"><span class="sw" data-key="{spec["key"]}" '
            f'style="background:rgba({r},{g},{b},{spec["opacity"] / 100})"></span>{spec["legend"]}</div>')


THEME_CSS = """
<style>
  :root { --bg:#0a0a0a; --raised:#131312; --ink:#f2f1ec; --ink2:#b5b4ad; --ink3:#8b8a84;
          --hair:#262624; --hair2:#3b3b38; --ut:0px; --ur:0px; --ub:0px; --ul:0px; }
  html, body { background: var(--bg); }
  .leaflet-container { background: var(--bg); color: var(--ink);
    font-family: 'Archivo', system-ui, sans-serif; }
  .dark-base { filter: brightness(0.36); }

  .lbl, #stats summary, #legend .t, .leaflet-control-layers label {
    text-transform: uppercase; font-size: 0.74rem; font-weight: 600; font-stretch: 108%;
    letter-spacing: 0.16em; font-variant-numeric: tabular-nums; }

  #stats, #legend, .leaflet-control-layers {
    background: rgba(10,10,10,0.86); -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
    border: 1px solid var(--hair); border-radius: 2px; box-shadow: none; color: var(--ink2);
    font-family: 'Archivo', system-ui, sans-serif; }
  #stats, #legend { position: fixed; z-index: 9999; font-size: 13px; line-height: 1.45; }
  #stats { top: calc(var(--ut) + 12px); left: calc(var(--ul) + 56px); max-width: 330px; padding: 10px 14px; }
  #stats summary { cursor: pointer; color: var(--ink); }
  #stats table { border-collapse: collapse; margin-top: 8px; width: 100%; }
  #stats td { padding: 3px 8px 3px 0; vertical-align: top; color: var(--ink2); }
  #stats td:last-child { color: var(--ink); text-align: right; padding-right: 0; font-variant-numeric: tabular-nums; }
  #stats .small { margin: 8px 0 0; font-size: 12px; color: var(--ink2); }
  #stats .small b { color: var(--ink); font-weight: 600; }
  #stats .note { margin: 8px 0 0; padding-top: 8px; border-top: 1px solid var(--hair);
    font-size: 11.5px; color: var(--ink3); }
  #stats .note b { color: var(--ink2); }
  #legend { bottom: calc(var(--ub) + 34px); left: calc(var(--ul) + 10px); padding: 10px 12px; font-size: 12px; }
  #legend .t { color: var(--ink); margin-bottom: 4px; font-size: 0.68rem; }
  #legend .t + .row, #legend .row + .row { margin-top: 3px; }
  .row { display: flex; align-items: center; gap: 8px; color: var(--ink2); }
  .sw { width: 12px; height: 12px; border-radius: 1px; display: inline-block; border: 1px solid rgba(255,255,255,0.18); }

  .leaflet-top { margin-top: var(--ut); } .leaflet-bottom { margin-bottom: var(--ub); }
  .leaflet-left { margin-left: var(--ul); } .leaflet-right { margin-right: var(--ur); }

  .leaflet-bar { border: 1px solid var(--hair2); box-shadow: none; border-radius: 2px; }
  .leaflet-bar a, .leaflet-bar a:hover, .leaflet-touch .leaflet-bar a {
    background-color: var(--raised); color: var(--ink); border-bottom: 1px solid var(--hair2); }
  .leaflet-bar a:hover { background-color: #1c1c1a; }
  .leaflet-bar a.leaflet-disabled { background-color: var(--raised); color: var(--hair2); }
  .leaflet-control-layers-toggle, .leaflet-retina .leaflet-control-layers-toggle {
    background-color: var(--raised) !important; border-radius: 2px;
    background-size: 20px 20px !important; background-repeat: no-repeat; background-position: center;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f2f1ec' stroke-width='1.6' stroke-linejoin='round'><path d='M12 3l9 5-9 5-9-5z'/><path d='M3 12.5l9 5 9-5'/><path d='M3 16.5l9 5 9-5'/></svg>") !important; }
  .leaflet-control-layers-expanded { padding: 8px 12px; color: var(--ink2); }
  .leaflet-control-layers-separator { border-top: 1px solid var(--hair); }
  .leaflet-control-layers label { color: var(--ink2); margin: 2px 0; font-size: 0.66rem; letter-spacing: 0.1em; }
  .leaflet-control-layers input { accent-color: #f2f1ec; }
  .leaflet-control-layers-selector { margin: 0 6px 0 0; }
  .leaflet-control-attribution { background: rgba(10,10,10,0.7) !important; color: var(--ink3) !important; }
  .leaflet-control-attribution a { color: var(--ink2) !important; }
  .leaflet-control-scale-line { color: var(--ink2); border: 1px solid var(--ink3); border-top: none;
    background: transparent; text-shadow: none; }

  @media (min-width: 601px) {
    .embed #stats { top: calc(var(--ut) + 24px); left: calc(var(--ul) + 24px); max-width: 360px; }
    .embed #legend { left: calc(var(--ul) + 24px); bottom: calc(var(--ub) + 50px); max-width: 360px; }
  }
  @media (max-width: 600px) {
    #stats { max-width: calc(100vw - 80px - var(--ul) - var(--ur)); top: calc(var(--ut) + 172px); }
    #legend { display: none; }
  }

  #trainnote { position: fixed; z-index: 9998; left: calc(var(--ul) + 250px); right: calc(var(--ur) + 300px);
    bottom: calc(var(--ub) + 8px); padding: 6px 12px; font-size: 11.5px; line-height: 1.4;
    color: var(--ink3); background: rgba(10,10,10,0.86); -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px); border: 1px solid var(--hair); border-radius: 2px;
    font-family: 'Archivo', system-ui, sans-serif; }
  #trainnote b { color: var(--ink2); font-weight: 600; }
  @media (max-width: 900px) { #trainnote { right: calc(var(--ur) + 200px); } }
  @media (max-width: 600px) { #trainnote { left: calc(var(--ul) + 8px); right: calc(var(--ur) + 8px);
    bottom: calc(var(--ub) + 34px); } }

  /* temporary colour test tool; hidden when embedded unless the hash has "colors" */
  #colortool { display: none; position: fixed; z-index: 9999; right: calc(var(--ur) + 10px);
    bottom: calc(var(--ub) + 34px); width: 300px; max-height: calc(100vh - 280px); overflow-y: auto;
    padding: 10px 12px; background: rgba(10,10,10,0.92); -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px); border: 1px solid var(--hair); border-radius: 2px; color: var(--ink2);
    font-family: 'Archivo', system-ui, sans-serif; font-size: 12px; }
  .colortool #colortool { display: block; }
  #colortool summary { cursor: pointer; color: var(--ink); text-transform: uppercase; font-size: 0.74rem;
    font-weight: 600; font-stretch: 108%; letter-spacing: 0.16em; }
  #colortool .ct-h { margin: 10px 0 4px; color: var(--ink); text-transform: uppercase; font-size: 0.68rem;
    font-weight: 600; letter-spacing: 0.16em; }
  #colortool .ct-row { display: grid; grid-template-columns: 1fr 30px 84px 38px; gap: 8px; align-items: center;
    margin: 4px 0; }
  #colortool input[type=color] { width: 30px; height: 20px; padding: 0; border: 1px solid var(--hair2);
    background: none; border-radius: 1px; cursor: pointer; }
  #colortool input[type=range] { width: 84px; accent-color: #b5b4ad; }
  #colortool .ct-val { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); }
  #colortool .ct-btns { display: flex; gap: 8px; margin: 10px 0 6px; }
  #colortool button { flex: 1; background: var(--raised); color: var(--ink); border: 1px solid var(--hair2);
    border-radius: 2px; padding: 5px 6px; font: inherit; cursor: pointer; }
  #colortool button:hover { background: #1c1c1a; }
  #colortool .ct-out { width: 100%; box-sizing: border-box; background: #0a0a0a; color: var(--ink2);
    border: 1px solid var(--hair); border-radius: 1px; font: 10.5px/1.35 ui-monospace, monospace; resize: vertical; }
  #colortool .ct-status { min-height: 1.2em; color: var(--ink3); font-size: 11px; }
  @media (max-width: 600px) { #colortool { left: 8px; right: 8px; width: auto; bottom: calc(var(--ub) + 90px); } }
</style>
<script>
  window.addEventListener('load', function () {
    // collapse the stats panel on phones, or when it would run into the legend
    var d = document.getElementById('stats'); if (!d) return;
    if (window.innerWidth < 600) { d.removeAttribute('open'); return; }
    var lg = document.getElementById('legend');
    if (lg && lg.getBoundingClientRect().top < d.getBoundingClientRect().bottom + 8) d.removeAttribute('open');
  });
</script>"""


EMBED_JS = """
<script>
(function () {
  var embed = window.parent !== window || /embed/.test(location.hash);
  if (!embed) return;
  var map = __MAP__, bounds = __BOUNDS__, city = __CITY__, years = __YEARS__;
  var baseDefault = __BASE_DEFAULT__, bases = [__BASES__];
  var overlays = [__OVERLAYS__];

  // Hash params, each "T,R,B,L" in CSS px. pad insets the fit box only; ui insets the
  // chrome (controls, panels, attribution) and defaults to pad when absent.
  function param(name) {
    var parts = location.hash.replace(/^#/, '').split('&');
    for (var i = 0; i < parts.length; i++) {
      var kv = parts[i].split('=');
      if (kv[0] === name && kv[1]) {
        var v = kv[1].split(',').map(parseFloat);
        if (v.length === 4 && v.every(function (x) { return !isNaN(x); })) return v;
      }
    }
    return null;
  }
  var pad = param('pad') || [0, 0, 0, 0];
  var ui = param('ui') || pad;
  var root = document.documentElement.style;
  root.setProperty('--ut', ui[0] + 'px'); root.setProperty('--ur', ui[1] + 'px');
  root.setProperty('--ub', ui[2] + 'px'); root.setProperty('--ul', ui[3] + 'px');
  document.documentElement.classList.add('embed');
  function placeZoom() {
    if (map.zoomControl) map.zoomControl.setPosition(window.innerWidth >= 600 ? 'topright' : 'topleft');
  }
  placeZoom();
  window.addEventListener('resize', placeZoom);
  map.options.zoomSnap = 0;

  function fit() {
    map.invalidateSize({ animate: false });
    map.fitBounds(bounds, { paddingTopLeft: [pad[3], pad[0]], paddingBottomRight: [pad[1], pad[2]], animate: false });
  }
  function overlayRect() {
    var nw = map.latLngToContainerPoint([bounds[1][0], bounds[0][1]]);
    var se = map.latLngToContainerPoint([bounds[0][0], bounds[1][1]]);
    return { left: nw.x, top: nw.y, width: se.x - nw.x, height: se.y - nw.y };
  }
  var sent = false, timer = null;
  function sendReady() {
    sent = true; clearTimeout(timer);
    window.parent.postMessage({ source: 'urbanlens-map', type: 'ready', city: city, years: years,
                                overlay: overlayRect() }, '*');
  }

  // Folium's own fitBounds starts an animated zoom, and Leaflet ignores setView
  // while one is running, so wait for it to end before fitting into the padded box.
  function runFit(done) {
    if (map._animatingZoom) {
      map.once('zoomend', function () { setTimeout(function () { fit(); done(); }, 0); });
    } else { fit(); done(); }
  }
  function whenBaseLoaded(done) {
    if (baseDefault._loading) baseDefault.once('load', done); else done();
  }

  runFit(function () { whenBaseLoaded(function () { if (!sent) sendReady(); }); });
  timer = setTimeout(function () { if (!sent) { fit(); sendReady(); } }, 4000);

  map.on('moveend', function () {
    if (sent) window.parent.postMessage({ source: 'urbanlens-map', type: 'view', overlay: overlayRect() }, '*');
  });

  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.source !== 'portfolio' || d.type !== 'reset') return;
    bases.forEach(function (b) { if (b !== baseDefault && map.hasLayer(b)) map.removeLayer(b); });
    if (!map.hasLayer(baseDefault)) baseDefault.addTo(map);
    overlays.forEach(function (o) {
      var on = map.hasLayer(o.layer);
      if (o.show && !on) o.layer.addTo(map);
      if (!o.show && on) map.removeLayer(o.layer);
    });
    runFit(sendReady);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') window.parent.postMessage({ source: 'urbanlens-map', type: 'escape' }, '*');
  });
})();
</script>"""


COLOR_TOOL_JS = """
<script>
(function () {
  var embed = window.parent !== window || /embed/.test(location.hash);
  // Temporary tool. Standalone pages show it. When embedded it stays hidden (and is not even
  // set up) unless: the iframe hash has "colors", the host page's own URL has "?colors" or
  // "#colors" (works when the host is same-origin), or Shift+C is pressed inside the map.
  function parentWants() {
    try { return window.parent !== window && /[?#&]colors/.test(window.parent.location.search + window.parent.location.hash); }
    catch (e) { return false; }
  }
  var started = false;
  function start() { if (!started) { started = true; setup(); } }
  if (!embed || /[?#&]colors/.test(location.hash) || parentWants()) start();
  else document.addEventListener('keydown', function (e) { if (e.shiftKey && e.key && e.key.toLowerCase() === 'c') start(); });

  function setup() {
  document.documentElement.classList.add('colortool');
  var specs = __SPECS__;
  var layerDefs = [__COLOR_LAYERS__];
  var STORE = 'urbanlens-colors-v2';
  var state = {}, defaults = {}, inputs = {};
  specs.forEach(function (s) { defaults[s.key] = { hex: s.hex, opacity: s.opacity }; state[s.key] = { hex: s.hex, opacity: s.opacity }; });
  try {
    var saved = JSON.parse(localStorage.getItem(STORE) || 'null');
    if (saved) specs.forEach(function (s) {
      var v = saved[s.key];
      if (v && /^#[0-9a-f]{6}$/i.test(v.hex) && v.opacity >= 0 && v.opacity <= 100) state[s.key] = { hex: v.hex, opacity: +v.opacity };
    });
  } catch (e) {}

  function rgb(hex) { return [parseInt(hex.substr(1, 2), 16), parseInt(hex.substr(3, 2), 16), parseInt(hex.substr(5, 2), 16)]; }
  function lutFor(group) {
    var lut = new Uint32Array(256);          // 0 = fully transparent
    specs.forEach(function (s) {
      if (s.group !== group) return;
      var c = rgb(state[s.key].hex), a = Math.round(255 * state[s.key].opacity / 100);
      lut[s.code] = ((a << 24) | (c[2] << 16) | (c[1] << 8) | c[0]) >>> 0;
    });
    return lut;
  }

  var prepared = layerDefs.map(function (def) { return { def: def, ready: false, dirty: false, timer: null, url: null }; });
  function draw(P) {
    var lut = lutFor(P.def.group), w = P.canvas.width, h = P.canvas.height;
    var out = P.ctx.createImageData(w, h), px = new Uint32Array(out.data.buffer);
    for (var i = 0; i < px.length; i++) px[i] = lut[P.idx[i]];
    P.ctx.putImageData(out, 0, 0);
    P.canvas.toBlob(function (blob) {
      var url = URL.createObjectURL(blob);
      P.def.layer.setUrl(url);
      if (P.url) URL.revokeObjectURL(P.url);
      P.url = url;
    }, 'image/png');
  }
  function schedule(P) {
    clearTimeout(P.timer);
    P.timer = setTimeout(function () { if (P.ready) draw(P); else P.dirty = true; }, 120);
  }
  function changedFromDefault(group) {
    return specs.some(function (s) {
      return s.group === group && (state[s.key].hex !== defaults[s.key].hex || state[s.key].opacity !== defaults[s.key].opacity);
    });
  }
  prepared.forEach(function (P) {
    var img = new Image();
    img.onload = function () {
      var c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
      var ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      var d = ctx.getImageData(0, 0, c.width, c.height).data, n = c.width * c.height, idx = new Uint8Array(n);
      for (var i = 0; i < n; i++) idx[i] = d[i * 4 + 3] === 255 ? d[i * 4] : 255;
      P.canvas = c; P.ctx = ctx; P.idx = idx; P.ready = true;
      if (P.dirty || changedFromDefault(P.def.group)) draw(P);
    };
    img.src = P.def.index;
  });

  var body = document.getElementById('colortool-body');
  var out = document.createElement('textarea'); out.className = 'ct-out'; out.readOnly = true; out.rows = 9;
  var status = document.createElement('div'); status.className = 'ct-status';
  function snapshot() {
    var o = {};
    specs.forEach(function (s) { o[s.key] = { color: state[s.key].hex, opacity: state[s.key].opacity }; });
    return o;
  }
  function refresh(s) {
    var c = rgb(state[s.key].hex);
    Array.prototype.forEach.call(document.querySelectorAll('[data-key="' + s.key + '"]'), function (el) {
      el.style.background = 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + state[s.key].opacity / 100 + ')';
    });
    inputs[s.key].col.value = state[s.key].hex;
    inputs[s.key].rng.value = state[s.key].opacity;
    inputs[s.key].val.textContent = state[s.key].opacity + '%';
  }
  function changed(s) {
    refresh(s);
    out.value = JSON.stringify(snapshot(), null, 1);
    try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {}
    prepared.forEach(function (P) { if (P.def.group === s.group) schedule(P); });
  }

  var titles = { change: 'Built-up change layer', lc: 'Land cover layers' };
  ['change', 'lc'].forEach(function (g) {
    var h = document.createElement('div'); h.className = 'ct-h'; h.textContent = titles[g]; body.appendChild(h);
    specs.filter(function (s) { return s.group === g; }).forEach(function (s) {
      var row = document.createElement('div'); row.className = 'ct-row';
      var name = document.createElement('span'); name.textContent = s.label;
      var col = document.createElement('input'); col.type = 'color';
      var rng = document.createElement('input'); rng.type = 'range'; rng.min = 0; rng.max = 100; rng.step = 1;
      var val = document.createElement('span'); val.className = 'ct-val';
      inputs[s.key] = { col: col, rng: rng, val: val };
      col.addEventListener('input', function () { state[s.key].hex = col.value; changed(s); });
      rng.addEventListener('input', function () { state[s.key].opacity = +rng.value; changed(s); });
      [name, col, rng, val].forEach(function (el) { row.appendChild(el); });
      body.appendChild(row);
      refresh(s);
    });
  });

  var btns = document.createElement('div'); btns.className = 'ct-btns';
  var log = document.createElement('button'); log.textContent = 'Log + copy';
  var reset = document.createElement('button'); reset.textContent = 'Reset';
  log.addEventListener('click', function () {
    var txt = JSON.stringify(snapshot(), null, 2);
    console.log('[urbanlens colors]', txt);
    out.select();
    var done = function () { status.textContent = 'Logged to the console and copied.'; };
    try { navigator.clipboard.writeText(txt).then(done, function () { document.execCommand('copy'); done(); }); }
    catch (e) { try { document.execCommand('copy'); } catch (e2) {} status.textContent = 'Logged to the console.'; }
  });
  reset.addEventListener('click', function () {
    specs.forEach(function (s) { state[s.key] = { hex: defaults[s.key].hex, opacity: defaults[s.key].opacity }; refresh(s); });
    out.value = JSON.stringify(snapshot(), null, 1);
    try { localStorage.removeItem(STORE); } catch (e) {}
    prepared.forEach(schedule);
    status.textContent = 'Reset to the default colours.';
  });
  btns.appendChild(log); btns.appendChild(reset);
  body.appendChild(btns); body.appendChild(out); body.appendChild(status);
  out.value = JSON.stringify(snapshot(), null, 1);
  }
})();
</script>"""


LEGEND_JS = """
<script>
(function () {
  var map = __MAP__, layers = [__LEGEND_LAYERS__];
  function update() {
    var on = {};
    layers.forEach(function (L) { if (map.hasLayer(L.layer)) on[L.group] = true; });
    Array.prototype.forEach.call(document.querySelectorAll('#legend [data-group]'), function (el) {
      el.style.display = on[el.getAttribute('data-group')] ? '' : 'none';
    });
    var lg = document.getElementById('legend');
    if (lg) lg.style.visibility = (on.change || on.lc) ? '' : 'hidden';
  }
  map.on('layeradd layerremove', update);
  update();
})();
</script>"""


def names_list(cities):
    names = [c.title() for c in cities]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def train_note(city, train_cities, test_cities):
    held = [f"<b>{c.title()}</b>" if c == city else c.title() for c in test_cities]
    held_txt = held[0] if len(held) == 1 else ", ".join(held[:-1]) + " and " + held[-1]
    verb = "was" if len(held) == 1 else "were"
    return (f"<b>Trained on:</b> {names_list(train_cities)}. "
            f"<b>Unseen during training:</b> {held_txt} {verb} held out and never seen by the model.")


def validation_sentence(city, channels):
    """Hand-label result for this city and model (Phase 5b), if scored."""
    if not VALIDATION_JSON.exists():
        return None
    res = json.loads(VALIDATION_JSON.read_text()).get(city)
    key = f"{channels}ch"
    if not res or key not in res["maps"]:
        return None
    oa, dw = res["maps"][key]["overall_accuracy"], res["maps"]["dynamic_world"]["overall_accuracy"]
    return (f"Checked against {res['n_points']} hand-labeled points in {city.title()} (2024 only), "
            f"it matched {100 * oa:.0f}% of them; Dynamic World matched {100 * dw:.0f}%.")


def build_panels(stats, city, year_a, year_b, specs, train_cities, test_cities, color_tool):
    m, d = stats["model"], stats["dynamic_world"]
    ua, ub = m["area_km2_date_a"]["Urban"], m["area_km2_date_b"]["Urban"]
    net = m["urban_net_change_km2"]
    src = m["urban_gain_by_source_km2"]
    total_gain = sum(src.values()) or 1.0
    sources = ", ".join(
        f"{k.lower()} {100 * v / total_gain:.0f}%"
        for k, v in sorted(src.items(), key=lambda kv: -kv[1]) if v / total_gain >= 0.005)
    channels = stats["in_channels"]
    checked = validation_sentence(city, channels)
    checked = (f"{checked} Change between years has not been checked by hand." if checked else
               "The figures show agreement with Dynamic World, not accuracy against ground truth.")

    stats_html = f"""
<details id="stats" open>
  <summary>{city.title()} / built-up change {year_a} to {year_b}</summary>
  <table>
    <tr><td>Built-up area {year_a}</td><td>{ua:,.0f} km&sup2;</td></tr>
    <tr><td>Built-up area {year_b}</td><td>{ub:,.0f} km&sup2;</td></tr>
    <tr><td>Net change</td><td>{net:+,.0f} km&sup2; ({100 * net / ua:+.1f}%)</td></tr>
    <tr><td>Gross gain / loss</td><td>{m['urban_gain_km2']:,.0f} / {m['urban_loss_km2']:,.0f} km&sup2;</td></tr>
    <tr><td>Dynamic World net</td><td>{d['urban_net_change_km2']:+,.0f} km&sup2;</td></tr>
  </table>
  <p class="small"><b>New built-up land came from:</b> {sources}.</p>
  <p class="note"><b>Read with care.</b> A {channels}-channel U-Net trained on Dynamic World
  labels classified two Sentinel-2 composites (Jan to Mar of each year, 10 m). {checked}
  Net change is the more reliable number. Built-up loss (not drawn) and part of the gross gain
  come from bare, dry ground read as built-up, not from demolition. Vegetation includes scrub
  and open land.</p>
</details>"""

    change_specs = [sp for sp in specs if sp["group"] == "change"]
    lc_specs = [sp for sp in specs if sp["group"] == "lc"]
    legend_html = f"""
<div id="legend">
  <div data-group="change">
    <div class="t">Built-up change</div>
    {''.join(swatch(sp) for sp in change_specs)}
  </div>
  <div data-group="lc" style="margin-top:8px">
    <div class="t">Land cover</div>
    {''.join(swatch(sp) for sp in lc_specs)}
  </div>
</div>"""

    note_html = f'<div id="trainnote">{train_note(city, train_cities, test_cities)}</div>'
    tool_html = ('<details id="colortool"><summary>Color test (temporary)</summary>'
                 '<div id="colortool-body"></div></details>') if color_tool else ""
    return THEME_CSS + stats_html + legend_html + note_html + tool_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--year-a", type=int, required=True)
    ap.add_argument("--year-b", type=int, required=True)
    ap.add_argument("--tag", default="unet5ch_v2_ablation_best",
                    help="checkpoint stem used when running change_detection.py")
    ap.add_argument("--max-dim", type=int, default=2000, help="longest side of each overlay, in pixels")
    ap.add_argument("--particles", action="store_true",
                    help="also write {city}_{a}_{b}_particles.bin/.json for the site's point cloud")
    ap.add_argument("--no-color-tool", action="store_true",
                    help="omit the temporary colour picker (smaller file); default is to include it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = load_config()
    class_names = config["classes"]["names"]
    train_cities, test_cities = config["cities"]["train"], config["cities"]["test"]
    specs = make_color_specs(class_names)
    color_tool = not args.no_color_tool
    pair = f"{args.city}_{args.year_a}_{args.year_b}_{args.tag}"
    stats = json.loads((CHANGE_DIR / f"{pair}_stats.json").read_text())
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    pred_a, bounds = to_web_mercator(CHANGE_DIR / f"{pair}_pred_{args.year_a}.tif", args.max_dim, Resampling.mode)
    pred_b, _ = to_web_mercator(CHANGE_DIR / f"{pair}_pred_{args.year_b}.tif", args.max_dim, Resampling.mode)

    (south, west), (north, east) = bounds
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=10,
                      tiles=None, control_scale=True, min_zoom=8, max_zoom=17)
    dark = folium.TileLayer(ESRI_DARK_GRAY, name="Dark gray (Esri)", max_native_zoom=16,
                            class_name="dark-base",
                            attr="Tiles &copy; Esri, DeLorme, NAVTEQ").add_to(fmap)
    sat = folium.TileLayer(ESRI_IMAGERY, name="Satellite imagery (Esri)", show=False,
                           attr="Tiles &copy; Esri, Maxar, Earthstar Geographics").add_to(fmap)
    osm = folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(fmap)

    # (name, per-pixel codes, colour group, visible by default). Opacity lives in each
    # colour's alpha, so the overlays themselves are fully opaque.
    layer_specs = [
        (f"Built-up change {args.year_a} to {args.year_b}", change_codes(pred_a, pred_b), "change", True),
        (f"Land cover {args.year_a}", pred_a, "lc", False),
        (f"Land cover {args.year_b}", pred_b, "lc", False),
    ]
    overlays, color_layers = [], []
    for name, codes, group, show in layer_specs:
        layer = ImageOverlay(image=png_data_uri(rgba_from_codes(codes, specs, group)), bounds=bounds,
                             name=name, opacity=1.0, show=show, interactive=False).add_to(fmap)
        overlays.append((layer, show))
        if color_tool:
            color_layers.append((layer, group, index_png_uri(codes)))

    # Layer list stays expanded (a collapsed toggle hid the layer names).
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(bounds)
    root = fmap.get_root()
    root.header.add_child(folium.Element(FONT_LINK))
    root.header.add_child(folium.Element(
        f"<title>{args.city.title()} urban change {args.year_a}-{args.year_b}</title>"))
    root.html.add_child(folium.Element(
        build_panels(stats, args.city, args.year_a, args.year_b, specs, train_cities, test_cities, color_tool)))

    embed_js = (EMBED_JS
                .replace("__MAP__", fmap.get_name())
                .replace("__BOUNDS__", json.dumps(bounds))
                .replace("__CITY__", json.dumps(args.city))
                .replace("__YEARS__", json.dumps([args.year_a, args.year_b]))
                .replace("__BASE_DEFAULT__", dark.get_name())
                .replace("__BASES__", ", ".join(b.get_name() for b in (dark, sat, osm)))
                .replace("__OVERLAYS__", ", ".join(
                    f"{{layer: {l.get_name()}, show: {str(sh).lower()}}}" for l, sh in overlays)))
    legend_js = LEGEND_JS.replace("__MAP__", fmap.get_name()).replace("__LEGEND_LAYERS__", ", ".join(
        f"{{layer: {l.get_name()}, group: {json.dumps(g)}}}"
        for (l, _), (_, _, g, _) in zip(overlays, layer_specs)))
    tool_js = ""
    if color_tool:
        tool_js = (COLOR_TOOL_JS
                   .replace("__SPECS__", json.dumps(specs))
                   .replace("__COLOR_LAYERS__", ", ".join(
                       f"{{layer: {l.get_name()}, group: {json.dumps(g)}, index: {json.dumps(uri)}}}"
                       for l, g, uri in color_layers)))

    # These scripts must run after Folium's own map script, so splice them in at the
    # end of the rendered document instead of adding them as children.
    html = root.render()
    head, sep, tail = html.rpartition("</html>")
    out = args.out or str(MAP_DIR / f"{args.city}_{args.year_a}_{args.year_b}_change_map.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(head + embed_js + legend_js + tool_js + sep + tail)
    print(f"saved -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")

    if args.particles:
        gw, gh, stem = write_particles(pred_a, pred_b, bounds, class_names,
                                       args.city, args.year_a, args.year_b, args.tag)
        print(f"particles -> {stem}.bin/.json  ({gw}x{gh} = {gw * gh} cells per date)")


if __name__ == "__main__":
    main()
