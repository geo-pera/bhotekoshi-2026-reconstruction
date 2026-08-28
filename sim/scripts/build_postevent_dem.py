"""Dense parallax dh surface -> post-event DEM estimate.

Dense tie points (32 px grid) between the opposite-look WV03 orthos; grid the
bias-corrected dh into a surface (median-of-neighbors, corridor-masked),
smooth, taper to zero outside measurement support, add to the pre-event DEM.
"""
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

tp = extract_tie_points(A, B, gtA, gtB, grid_res=32, window_size=96,
                        max_shift=100, min_reliability=12.0, nodata=0,
                        ransac=False, min_std=1.2, min_range_fraction=0.002)
n = len(tp["x_map"]); print(f"tie points: {n}", flush=True)
x, y = tp["x_map"], tp["y_map"]
along = tp["x_shift_m"] * np.sin(AZ) + tp["y_shift_m"] * np.cos(AZ)
rel = tp["reliability"]
keep = rel >= 12
x, y, along = x[keep], y[keep], along[keep]
print(f"reliable: {keep.sum()}", flush=True)

# bias plane from points >500 m from centerline
cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
cx = np.array([float(r["x_utm45"]) for r in cl]); cy = np.array([float(r["y_utm45"]) for r in cl])
d2 = (x[:, None] - cx[None, ::5]) ** 2 + (y[:, None] - cy[None, ::5]) ** 2
dist = np.sqrt(d2.min(axis=1))
stab = dist > 500
Ab = np.column_stack([np.ones(stab.sum()), x[stab] - x.mean(), y[stab] - y.mean()])
coef, *_ = np.linalg.lstsq(Ab, along[stab], rcond=None)
dh = (along - (coef[0] + coef[1] * (x - x.mean()) + coef[2] * (y - y.mean()))) / TANSUM
print(f"bias const {coef[0]:+.2f}; corridor pts (<400m): {(dist<400).sum()}", flush=True)

# grid dh at 32 m over the overlap: median in cells, then 3x3 median, light smooth
RES = 32.0
x0, y1 = gtA[0], gtA[3]
NXg = int(A.shape[1] * 1.0 / RES); NYg = int(A.shape[0] * 1.0 / RES)
gi = ((y1 - y) / RES).astype(int); gj = ((x - x0) / RES).astype(int)
ok = (gi >= 0) & (gi < NYg) & (gj >= 0) & (gj < NXg) & (dist < 600)   # corridor support only
sum_ = np.zeros((NYg, NXg)); cnt = np.zeros((NYg, NXg))
np.add.at(sum_, (gi[ok], gj[ok]), dh[ok]); np.add.at(cnt, (gi[ok], gj[ok]), 1)
grid = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan)

def boxmed(a, k=1):
    stack = []
    for di in range(-k, k+1):
        for dj in range(-k, k+1):
            stack.append(np.roll(np.roll(a, di, 0), dj, 1))
    return np.nanmedian(np.stack(stack), axis=0)
grid = boxmed(grid, 1)          # fills small holes, suppresses outliers
grid = boxmed(grid, 1)
supp = np.isfinite(grid)
print(f"dh surface support: {supp.sum()} cells ({supp.sum()*RES*RES/1e6:.1f} km2), "
      f"median {np.nanmedian(grid):+.1f} m", flush=True)
gridf = np.where(supp, grid, 0.0)
# taper: zero beyond support, feather 2 cells
feather = supp.astype(float)
for _ in range(2):
    feather = 0.25 * (np.roll(feather,1,0)+np.roll(feather,-1,0)+np.roll(feather,1,1)+np.roll(feather,-1,1))
gridf *= np.clip(feather, 0, 1)
# clamp physical range
gridf = np.clip(gridf, -30, 40)

# write dh surface + updated DEM patch (whole-domain DEM updated where overlap)
from rasterio.transform import from_origin
tr32 = from_origin(x0, y1, RES, RES)
with rasterio.open(f"{ROOT}/sim/dem/dh_surface_32m.tif", "w", driver="GTiff",
                   width=NXg, height=NYg, count=1, dtype="float32", crs="EPSG:32645",
                   transform=tr32, nodata=-9999, compress="deflate") as dst:
    dst.write(np.where(supp, grid, -9999).astype(np.float32), 1)
np.save(f"{ROOT}/sim/dem/dh_fill_32m.npy", gridf.astype(np.float32))
with open(f"{ROOT}/sim/dem/dh_meta.txt", "w") as f:
    f.write(f"{x0} {y1} {RES} {NXg} {NYg}\n")
print("wrote dh_surface_32m.tif + fill grid", flush=True)
