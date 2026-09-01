"""Vectorise the raster analysis into GIS-ready GeoJSON layers (EPSG:4326).

Outputs to vectors/. Layers:
  flood_extent_event            modelled peak inundation, 26 Aug event (8 m run)
  flood_extent_breach_scenario  modelled breach-scenario inundation (16 m run)
  inundation_probability_2m     ensemble P(depth>2 m) classes 0.1/0.5/0.9
  deposition_wedge              stereo-measured elevation gain > +4 m
  erosion_zones                 stereo-measured elevation loss < -4 m
  sar_new_dark_patches          Sentinel-1 16->28 Aug new-dark candidates
  trimline_observations         bank flow-height points (chainage-projected)
  superelevation_velocities     bend-velocity points
  settlement_arrival_times      modelled wave arrival per settlement
  river_centerline / km_markers flow path with chainage
  lower_trishuli_disturbance    day-2 stripped-vegetation polygons (km 72-95)
  buildings_at_risk_lower_trishuli  OSM buildings in/near detected scour

Run with a python that has rasterio+shapely (the sim venv works).
"""
import csv, json, os
import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform_geom, transform as rtf
from rasterio.transform import Affine
from scipy import ndimage

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT = f"{ROOT}/vectors"
os.makedirs(OUT, exist_ok=True)
UTM = "EPSG:32645"

def write_gj(name, feats):
    fc = {"type": "FeatureCollection", "features": feats}
    with open(f"{OUT}/{name}.geojson", "w") as f:
        json.dump(fc, f)
    print(f"{name}: {len(feats)} features")

def mask_to_feats(mask, transform, props, min_px=8, opening=True):
    m = ndimage.binary_opening(mask, iterations=1) if opening else mask
    m = features.sieve(m.astype(np.uint8), min_px)
    feats = []
    for geom, val in features.shapes(m, mask=m.astype(bool), transform=transform):
        g4326 = transform_geom(UTM, "EPSG:4326", geom, precision=6)
        feats.append({"type": "Feature", "geometry": g4326, "properties": dict(props)})
    return feats

def gt_to_affine(gt):
    return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])

# --- centerline helpers ---
cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
ch = np.array([float(c["chainage_m"]) for c in cl])
lon_cl = np.array([float(c["lon"]) for c in cl])
lat_cl = np.array([float(c["lat"]) for c in cl])
def at_chainage(c_m):
    return float(np.interp(c_m, ch, lon_cl)), float(np.interp(c_m, ch, lat_cl))

# 1. event peak extent (8 m run: depth = peak_eta - Z)
d = np.load(f"{ROOT}/sim/runs/swe2d_gorge_8m_postDEM.npz")
depth = np.where(d["peak_eta"] > -1e29, d["peak_eta"] - d["Z"], 0)
write_gj("flood_extent_event",
         mask_to_feats(depth >= 0.5, gt_to_affine(d["gt"]),
                       {"layer": "event peak inundation", "depth_min_m": 0.5,
                        "model": "2D SWE 8 m, V=100 Mm3"}, min_px=40))

# 2. breach scenario extent
d = np.load(f"{ROOT}/sim/runs/swe2d_lende_breach_gpu2.npz")
write_gj("flood_extent_breach_scenario",
         mask_to_feats(d["peak_depth"] >= 0.25, gt_to_affine(d["gt"]),
                       {"layer": "breach scenario inundation", "volume_Mm3": 5,
                        "depth_min_m": 0.25}, min_px=20))

# 3. inundation probability classes (ensemble, depth > 2 m)
d = np.load(f"{ROOT}/sim/runs/ensemble_2d.npz")
P = d["counts"][1].astype(np.float32) / int(d["n_run"])
tr = gt_to_affine(d["gt"])
feats = []
for lo, label in [(0.1, "P>=0.1"), (0.5, "P>=0.5"), (0.9, "P>=0.9")]:
    feats += mask_to_feats(P >= lo, tr, {"prob_class": label, "depth_m": 2.0,
                                         "members": int(d["n_run"])}, min_px=8)
write_gj("inundation_probability_2m", feats)

# 4/5. deposition and erosion from the stereo-DSM elevation change.
# Supersedes the retracted ortho-parallax layers (see README "Correction").
# The dh rasters ship as release assets: dh_reachA_2m.tif / dh_reachB_2m.tif.
import glob as _glob
for _p in sorted(_glob.glob(f"{ROOT}/vantor/stereo/release/dh_reach?_2m.tif")):
    _reach = _p.split("dh_reach")[1][0]
    with rasterio.open(_p) as s:
        dh = s.read(1, masked=True).filled(np.nan)
        dtr = s.transform
    dh[np.abs(dh) > 60] = np.nan
    write_gj(f"deposition_dsm_reach{_reach}",
             mask_to_feats(np.nan_to_num(dh) > 2, dtr, {"layer": "deposition > +2 m",
                           "method": "0.5 m stereo DSM minus pre-event DEM"}, min_px=60))
    write_gj(f"erosion_dsm_reach{_reach}",
             mask_to_feats(np.nan_to_num(dh) < -2, dtr, {"layer": "erosion < -2 m",
                           "method": "0.5 m stereo DSM minus pre-event DEM"}, min_px=60))

# 6. SAR new-dark patches
d = np.load(f"{ROOT}/sim/inputs/s1_amplitude_change.npz")
cand = d["cand"]
tr = Affine(20.0, 0, float(d["x0"]), 0, -20.0, float(d["y1"]))
lab, n = ndimage.label(cand)
feats = []
for i in range(1, n + 1):
    m = lab == i
    if m.sum() < 8:
        continue
    for geom, _ in features.shapes(m.astype(np.uint8), mask=m, transform=tr):
        g = transform_geom(UTM, "EPSG:4326", geom, precision=6)
        feats.append({"type": "Feature", "geometry": g,
                      "properties": {"area_m2": int(m.sum()) * 400,
                                     "layer": "S1 VV new-dark 16->28 Aug",
                                     "note": "candidate surface change; cluster near "
                                             "85.564E 28.266N interpreted as detachment zone"}})
write_gj("sar_new_dark_patches", feats)

# 7. trimline observations
feats = []
for row in csv.DictReader(open(f"{ROOT}/sim/inputs/trimline_profile_v2.csv")):
    c = float(row["chainage_m"])
    lon, lat = at_chainage(c)
    for side in ("L", "R"):
        try:
            h = float(row[f"h_{side}_m"])
        except (ValueError, TypeError, KeyError):
            continue
        if np.isfinite(h) and h > 0.5:
            quality = "suspect_haze_artifact" if 30500 <= c <= 33000 else "ok"
            feats.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                          "properties": {"chainage_km": round(c / 1000, 2), "bank": side,
                                         "flow_height_m": round(h, 1),
                                         "quality": quality,
                                         "uncertain": bool(int(row.get(f"void_{side}", 0) or 0))}})
write_gj("trimline_observations", feats)

# 8. superelevation velocities
feats = []
for row in csv.DictReader(open(f"{ROOT}/sim/inputs/superelevation_velocities.csv")):
    c = float(row["chainage_m"])
    lon, lat = at_chainage(c)
    feats.append({"type": "Feature",
                  "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                  "properties": {k: (float(v) if v.replace('.', '', 1).replace('-', '', 1).isdigit() else v)
                                 for k, v in row.items()}})
write_gj("superelevation_velocities", feats)

# 9. settlement arrival times
feats = []
for row in csv.DictReader(open(f"{ROOT}/sim/runs/arrival_times.csv")):
    lon, lat = at_chainage(float(row["chainage_km"]) * 1000)
    feats.append({"type": "Feature",
                  "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                  "properties": dict(row)})
write_gj("settlement_arrival_times", feats)

# 10. centerline + km markers
line = [[round(float(lo), 6), round(float(la), 6)] for lo, la in zip(lon_cl, lat_cl)]
write_gj("river_centerline",
         [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": line},
           "properties": {"layer": "flow path source->past Galchhi", "length_km": round(ch[-1] / 1000, 1)}}])
feats = []
for km in range(0, int(ch[-1] / 1000) + 1, 5):
    lon, lat = at_chainage(km * 1000)
    feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                  "properties": {"chainage_km": km}})
write_gj("km_markers", feats)

# 11. lower Trishuli disturbance
d = np.load(f"{ROOT}/sim/inputs/lower_trishuli_disturbance.npz")
tr = Affine(float(d["res"]), 0, float(d["x0"]), 0, -float(d["res"]), float(d["y1"]))
write_gj("lower_trishuli_disturbance",
         mask_to_feats(d["dist"], tr, {"layer": "stripped vegetation, day 2",
                       "note": "cloud-free 59% of corridor only"}, min_px=8))

# 12. buildings at risk (lower Trishuli)
bj = json.load(open(f"{ROOT}/sim/inputs/osm_buildings_lower_trishuli.json"))
els = [e for e in bj.get("elements", []) if "center" in e]
dist, cloud = d["dist"], d["cloud"]
dist50 = ndimage.binary_dilation(dist, iterations=int(50 / float(d["res"])))
H, W = dist.shape
inv = ~tr
feats = []
X, Y = rtf("EPSG:4326", UTM, [e["center"]["lon"] for e in els], [e["center"]["lat"] for e in els])
for e, x, y in zip(els, X, Y):
    cjj, rii = inv * (x, y)
    i, j = int(rii), int(cjj)
    if not (0 <= i < H and 0 <= j < W):
        continue
    if dist[i, j]:
        cls = "in_detected_scour"
    elif dist50[i, j]:
        cls = "within_50m_of_scour"
    else:
        continue
    feats.append({"type": "Feature",
                  "geometry": {"type": "Point",
                               "coordinates": [round(e["center"]["lon"], 6), round(e["center"]["lat"], 6)]},
                  "properties": {"osm_id": e.get("id"), "class": cls}})
write_gj("buildings_at_risk_lower_trishuli", feats)

print("ALL VECTOR LAYERS WRITTEN to", OUT)
