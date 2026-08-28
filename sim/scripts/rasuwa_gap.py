"""Bare-corridor edge heights for the Rasuwagadhi reach (km 18-33).

The July 2025 flood pre-stripped this braidplain, so veg-loss disturbance is
blind here. Instead: channel-connected bare&bright (or water) corridor in the
POST scenes vs the same in PRE scenes (2025 baseline). Edge heights above
thalweg for both; where post > pre the 2026 event exceeded the 2025 scour.
"""
import glob, csv
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
g = gdal.Open(f"{ROOT}/sim/dem/domain_8m_filled_ortho.tif")
GT = g.GetGeoTransform(); NX, NY = g.RasterXSize, g.RasterYSize
DEM = g.GetRasterBand(1).ReadAsArray()

pre_clear = np.zeros((NY, NX), bool); pre_bare = np.zeros((NY, NX), bool); pre_veg = np.zeros((NY, NX), bool)
post_clear = np.zeros((NY, NX), bool); post_bare = np.zeros((NY, NX), bool); post_veg = np.zeros((NY, NX), bool)
for f in sorted(glob.glob(f"{ROOT}/sim/masks/scenebits_*.tif")):
    ds = gdal.Open(f); bits = ds.GetRasterBand(1).ReadAsArray()
    clear = (bits & 1) > 0
    veg = (bits & 2) > 0
    bare = clear & ~veg & (((bits & 4) > 0) | ((bits & 8) > 0))
    if "_202605" in f:
        pre_clear |= clear; pre_bare |= bare; pre_veg |= veg
    else:
        post_clear |= clear; post_bare |= bare; post_veg |= veg
pre_bare &= ~pre_veg
post_bare &= ~post_veg

cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
ch = np.array([float(r["chainage_m"]) for r in cl])
xs = np.array([float(r["x_utm45"]) for r in cl]); ys = np.array([float(r["y_utm45"]) for r in cl])
seed = np.zeros((NY, NX), bool)
for x, y in zip(xs, ys):
    c = int((x - GT[0]) / GT[1]); r = int((y - GT[3]) / GT[5])
    if 0 <= r < NY and 0 <= c < NX: seed[max(0, r-13):r+13, max(0, c-13):c+13] = True

def connect(mask):
    out = np.zeros_like(mask); frontier = mask & seed
    while frontier.any():
        out |= frontier
        grow = np.zeros_like(mask)
        grow[1:, :] |= frontier[:-1, :]; grow[:-1, :] |= frontier[1:, :]
        grow[:, 1:] |= frontier[:, :-1]; grow[:, :-1] |= frontier[:, 1:]
        frontier = grow & mask & ~out
    return out
pre_c = connect(pre_bare); post_c = connect(post_bare)

def sample(arr, x, y):
    c = int((x - GT[0]) / GT[1]); r = int((y - GT[3]) / GT[5])
    if 0 <= r < NY and 0 <= c < NX: return arr[r, c]
    return False

def edge_height(mask, obs, x0, y0, nxv, nyv, sign, thal):
    run_max = np.nan; gap = 0; started = False
    for dd in np.arange(0, 401, 8.0):
        x, y = x0 + sign * nxv * dd, y0 + sign * nyv * dd
        if not sample(obs, x, y):
            gap += 1
            if started and gap > 3: break
            continue
        if sample(mask, x, y):
            started = True; gap = 0
            c = int((x - GT[0]) / GT[1]); r = int((y - GT[3]) / GT[5])
            z = DEM[r, c]
            if z > -9000: run_max = z if np.isnan(run_max) else max(run_max, z)
        else:
            if started:
                gap += 1
                if gap > 3: break
    return run_max - thal if np.isfinite(run_max) else np.nan

print(f"{'km':>6} {'side':>4} {'pre2025(m)':>10} {'post2026(m)':>11} {'delta':>6}")
res = []
for s in np.arange(18000, 33000, 400.0):
    i = np.clip(np.searchsorted(ch, s), 5, len(ch) - 6)
    x0 = np.interp(s, ch, xs); y0 = np.interp(s, ch, ys)
    dx = xs[i+5] - xs[i-5]; dy = ys[i+5] - ys[i-5]; L = np.hypot(dx, dy) or 1.0
    nxv, nyv = -dy / L, dx / L
    tv = [DEM[int((y0 + nyv * dd - GT[3]) / GT[5]), int((x0 + nxv * dd - GT[0]) / GT[1])]
          for dd in np.arange(-96, 97, 8.0)]
    tv = [z for z in tv if z > -9000]
    if not tv: continue
    thal = min(tv)
    for side, sign in (("L", 1.0), ("R", -1.0)):
        hpre = edge_height(pre_c, pre_clear, x0, y0, nxv, nyv, sign, thal)
        hpost = edge_height(post_c, post_clear, x0, y0, nxv, nyv, sign, thal)
        if np.isfinite(hpost):
            dtx = f"{hpost-hpre:+6.1f}" if np.isfinite(hpre) else "     ?"
            print(f"{s/1000:6.1f} {side:>4} {hpre if np.isfinite(hpre) else float('nan'):10.1f} {hpost:11.1f} {dtx}")
            res.append((s, side, hpre, hpost))
post_all = [r[3] for r in res]
if post_all:
    print(f"\nstations with post-event edge: {len(post_all)}, median height {np.median(post_all):.1f} m, "
          f"p90 {np.percentile(post_all, 90):.1f} m")
with open(f"{ROOT}/sim/inputs/rasuwa_bare_edge.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["chainage_m", "side", "h_pre2025_m", "h_post2026_m"])
    for s, side, hp, ho in res: w.writerow([f"{s:.0f}", side, f"{hp:.1f}", f"{ho:.1f}"])
print("wrote sim/inputs/rasuwa_bare_edge.csv")
