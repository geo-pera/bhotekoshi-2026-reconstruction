"""Dense elevation-change from WV03 opposite-look ortho parallax,
using a dense phase-correlation tie-point engine (user-supplied)."""
import sys, csv
import numpy as np
import rasterio
# Dense tie-point matching requires an external phase-correlation engine
# (not included). Any implementation with the interface below works:
#   extract_tie_points(ref, tgt, ref_gt, tgt_gt, grid_res, window_size,
#   max_shift, min_reliability, nodata, ransac, min_std[, min_range_fraction])
#   -> dict(x_map, y_map, x_shift_m, y_shift_m, reliability, inlier)
from tie_points import extract_tie_points  # user-supplied module

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
TANSUM = np.tan(np.radians(21.72)) + np.tan(np.radians(26.56))
AZ = np.radians(14.3)

with rasterio.open(f"{ROOT}/vantor/work/B040001100881410_g_utm.tif") as s:
    A = s.read(1); gtA = s.transform.to_gdal()
with rasterio.open(f"{ROOT}/vantor/work/B040001100881710_g_utm.tif") as s:
    B = s.read(1); gtB = s.transform.to_gdal()
print(f"arrays {A.shape}, px {gtA[1]} m", flush=True)

tp = extract_tie_points(A, B, gtA, gtB, grid_res=120, window_size=128,
                        max_shift=120, min_reliability=25.0, nodata=0,
                        ransac=False, min_std=4.0)
n = len(tp["x_map"])
print(f"tie points: {n}", flush=True)
if n == 0: sys.exit(1)
x, y = tp["x_map"], tp["y_map"]
sx, sy = tp["x_shift_m"], tp["y_shift_m"]
rel = tp["reliability"]
along = sx * np.sin(AZ) + sy * np.cos(AZ)
keep = rel >= 25.0
print(f"reliable: {keep.sum()}", flush=True)
x, y, along, rel = x[keep], y[keep], along[keep], rel[keep]

# distance to centerline for stable/corridor split
cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
cx = np.array([float(r["x_utm45"]) for r in cl]); cy = np.array([float(r["y_utm45"]) for r in cl])
ch = np.array([float(r["chainage_m"]) for r in cl])
d2 = ((x[:, None] - cx[None, ::4]) ** 2 + (y[:, None] - cy[None, ::4]) ** 2)
imin = np.argmin(d2, axis=1)
dist = np.sqrt(d2[np.arange(len(x)), imin])
chain = ch[::4][imin]

stable = dist > 500
print(f"stable points: {stable.sum()}, corridor(<300m): {(dist<300).sum()}", flush=True)
# bias: linear surface fit on stable points (removes residual ortho misregistration ramps)
Ab = np.column_stack([np.ones(stable.sum()), x[stable] - x.mean(), y[stable] - y.mean()])
coef, *_ = np.linalg.lstsq(Ab, along[stable], rcond=None)
resid = along[stable] - Ab @ coef
print(f"bias plane: const {coef[0]:+.2f} m, resid std {resid.std():.2f} m", flush=True)
bias_all = coef[0] + coef[1] * (x - x.mean()) + coef[2] * (y - y.mean())
dh = (along - bias_all) / TANSUM

with open(f"{ROOT}/sim/inputs/stereo_dh_dense.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["x", "y", "chainage_m", "dist_m", "dh_m", "reliability"])
    for i in range(len(x)):
        w.writerow([f"{x[i]:.0f}", f"{y[i]:.0f}", f"{chain[i]:.0f}", f"{dist[i]:.0f}",
                    f"{dh[i]:.2f}", f"{rel[i]:.0f}"])
cor = dist < 300
print(f"\ncorridor dh: n={cor.sum()}, median {np.median(dh[cor]):+.1f} m, "
      f"p10 {np.percentile(dh[cor],10):+.1f}, p90 {np.percentile(dh[cor],90):+.1f}")
print(f"stable dh (should be ~0): median {np.median(dh[stable]):+.1f}, std {dh[stable].std():.1f}")
# profile
print(f"{'km':>5} {'n':>4} {'median dh':>10}")
for k in range(26, 40):
    m = cor & (chain >= k*1000) & (chain < (k+1)*1000)
    if m.sum() >= 3:
        print(f"{k:5d} {m.sum():4d} {np.median(dh[m]):+10.1f}")
