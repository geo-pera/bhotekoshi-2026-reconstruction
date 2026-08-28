"""Cross-section hypsometry for the 1D routing model.

Every DX_ST m along centerline v3: perpendicular section, walk outward with a
monotone elevation envelope (pockets beyond a ridge crest don't count until the
stage tops the crest). Tables: stage above thalweg (0..SMAX step 1 m) -> wetted
area A (m^2) and top width B (m). Saved to sim/inputs/xsections.npz
"""
import csv, sys
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
DX_ST = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
HALF = 800.0     # half section length (m)
DS = 8.0         # sample step along section (m)
SMAX = 120      # max stage above thalweg (m)

import os
dem_ds = gdal.Open(os.environ.get("DEM", f"{ROOT}/sim/dem/domain_8m_filled_ortho.tif"))
SUFFIX = os.environ.get("XS_SUFFIX", "")
GT = dem_ds.GetGeoTransform()
DEM = dem_ds.GetRasterBand(1).ReadAsArray()
NY, NX = DEM.shape

cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
ch = np.array([float(r["chainage_m"]) for r in cl])
xs = np.array([float(r["x_utm45"]) for r in cl])
ys = np.array([float(r["y_utm45"]) for r in cl])

def zat(x, y):
    c = int((x - GT[0]) / GT[1]); r = int((y - GT[3]) / GT[5])
    if 0 <= r < NY and 0 <= c < NX:
        z = DEM[r, c]
        return z if z > -9000 else np.nan
    return np.nan

stations = np.arange(0.0, ch[-1] - DX_ST, DX_ST)
stage = np.arange(0, SMAX + 1, 1.0)
A_tab = np.zeros((len(stations), len(stage)), np.float32)
B_tab = np.zeros((len(stations), len(stage)), np.float32)
thal = np.zeros(len(stations), np.float32)
sx_arr = np.zeros(len(stations)); sy_arr = np.zeros(len(stations))

for k, s in enumerate(stations):
    i = np.clip(np.searchsorted(ch, s), 5, len(ch) - 6)
    x0 = np.interp(s, ch, xs); y0 = np.interp(s, ch, ys)
    dx = xs[i+5] - xs[i-5]; dy = ys[i+5] - ys[i-5]
    L = np.hypot(dx, dy)
    if L == 0: L = 1.0
    nxv, nyv = -dy / L, dx / L
    # thalweg: min elevation within +-100 m of the centerline point
    zc = [zat(x0 + nxv * d, y0 + nyv * d) for d in np.arange(-96, 97, DS)]
    zc = [z for z in zc if np.isfinite(z)]
    t = min(zc) if zc else np.nan
    thal[k] = t; sx_arr[k] = x0; sy_arr[k] = y0
    if not np.isfinite(t): continue
    # outward walks with monotone envelope, relative depth profile
    prof = []
    for sign in (1.0, -1.0):
        env = -np.inf
        for d in np.arange(0, HALF + 1, DS):
            z = zat(x0 + sign * nxv * d, y0 + sign * nyv * d)
            if not np.isfinite(z): z = env if np.isfinite(env) else t
            env = max(env, z)
            prof.append(env - t)
            if env - t > SMAX + 5: break
    p = np.array(prof)
    for j, st in enumerate(stage):
        wet = p < st
        B_tab[k, j] = wet.sum() * DS
        A_tab[k, j] = np.sum(np.clip(st - p[wet], 0, None)) * DS

# clean thalweg: interpolate gaps, enforce non-increasing downstream, light smooth
ok = np.isfinite(thal)
thal = np.interp(stations, stations[ok], thal[ok]).astype(np.float32)
thal_mono = np.minimum.accumulate(thal)
MINSLOPE = 0.0005
for i in range(1, len(thal_mono)):
    thal_mono[i] = min(thal_mono[i], thal_mono[i-1] - MINSLOPE * DX_ST)
kern = np.ones(5) / 5
thal_s = np.convolve(np.pad(thal_mono, 2, mode='edge'), kern, mode='valid').astype(np.float32)

np.savez_compressed(f"{ROOT}/sim/inputs/xsections.npz" if DX_ST == 200 else f"{ROOT}/sim/inputs/xsections{SUFFIX}_{DX_ST:.0f}.npz",
                    chainage=stations, thalweg=thal_s, thalweg_raw=thal,
                    stage=stage, A=A_tab, B=B_tab, x=sx_arr, y=sy_arr)
print(f"{len(stations)} sections, chainage 0..{stations[-1]/1000:.1f} km")
print(f"thalweg {thal_s[0]:.0f} -> {thal_s[-1]:.0f} m")
slope = -np.gradient(thal_s, stations)
print(f"bed slope: median {np.median(slope):.4f}, upper-reach (0-22km) {np.median(slope[stations<22000]):.4f}, "
      f"lower (70-120km) {np.median(slope[stations>70000]):.4f}")
for st_q in (10, 30, 60):
    j = int(st_q)
    print(f"width at +{st_q} m stage: median {np.median(B_tab[:, j]):.0f} m, "
          f"p10 {np.percentile(B_tab[:, j], 10):.0f}, p90 {np.percentile(B_tab[:, j], 90):.0f}")
