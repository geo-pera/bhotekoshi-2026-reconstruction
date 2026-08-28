"""Trimline height extraction along the centerline.

At each station: cross-section perpendicular to the smoothed centerline,
walk outward from the channel on each bank through contiguous disturbed pixels
(disturbance.tif class 1, gaps <= GAP px tolerated), take the max DEM elevation
of that run as the trimline; height = trimline - thalweg.

Writes sim/inputs/trimline_profile.csv
"""
import csv
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
STEP = 200.0        # station spacing (m)
HALF = 400.0        # half cross-section length (m)
GAP = 3             # tolerated gap in disturbed run (pixels)
THALWEG_HALF = 100  # m around center for thalweg search

dem_ds = gdal.Open(f"{ROOT}/sim/dem/domain_8m_filled_ortho.tif")
GT = dem_ds.GetGeoTransform()
DEM = dem_ds.GetRasterBand(1).ReadAsArray()
import os
_d = gdal.Open(os.environ.get("DIST", f"{ROOT}/sim/masks/disturbance.tif"))
DIST = _d.GetRasterBand(1).ReadAsArray()
NY, NX = DEM.shape

# HMA validity (void => GLO fill => higher uncertainty)
_h = gdal.Open(f"{ROOT}/sim/dem/domain_8m_utm45.tif")
HMA = _h.GetRasterBand(1).ReadAsArray()
VOID = (HMA <= -9000)
del HMA

cl = [(float(r["chainage_m"]), float(r["x_utm45"]), float(r["y_utm45"]))
      for r in csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv"))]
ch = np.array([c[0] for c in cl]); xs = np.array([c[1] for c in cl]); ys = np.array([c[2] for c in cl])

def sample(x, y):
    c = int((x - GT[0]) / GT[1]); r = int((y - GT[3]) / GT[5])
    if 0 <= r < NY and 0 <= c < NX:
        return DEM[r, c], DIST[r, c], VOID[r, c]
    return np.nan, 255, True

out = []
stations = np.arange(0, ch[-1], STEP)
for s in stations:
    i = np.searchsorted(ch, s)
    if i < 5 or i > len(ch) - 6: continue
    x0 = np.interp(s, ch, xs); y0 = np.interp(s, ch, ys)
    # smoothed direction over +-10 vertices
    dx = xs[i+5] - xs[i-5]; dy = ys[i+5] - ys[i-5]
    L = np.hypot(dx, dy)
    if L == 0: continue
    nxv, nyv = -dy / L, dx / L   # unit normal
    # thalweg: min elev within +-THALWEG_HALF of center along the section
    tvals = []
    for d in np.arange(-THALWEG_HALF, THALWEG_HALF + 1, 8.0):
        z, _, _ = sample(x0 + nxv * d, y0 + nyv * d)
        if np.isfinite(z): tvals.append(z)
    if not tvals: continue
    thal = min(tvals)
    rec = {"chainage_m": s, "x": x0, "y": y0, "thalweg_m": thal}
    for side, sign in (("L", 1.0), ("R", -1.0)):
        run_max = np.nan; gap = 0; started = False; nobs = 0; void_in_run = False
        d_end = np.nan
        for d in np.arange(0, HALF + 1, 8.0):
            z, dcls, v = sample(x0 + sign * nxv * d, y0 + sign * nyv * d)
            if dcls == 255:
                gap += 1
                if started and gap > GAP: break
                continue
            nobs += 1
            if dcls == 1:
                started = True; gap = 0; d_end = d
                if np.isfinite(z):
                    run_max = z if np.isnan(run_max) else max(run_max, z)
                    void_in_run |= v
            else:
                if started:
                    gap += 1
                    if gap > GAP: break
        rec[f"trim_{side}_m"] = run_max
        rec[f"h_{side}_m"] = run_max - thal if np.isfinite(run_max) else np.nan
        rec[f"d_{side}_m"] = d_end
        rec[f"void_{side}"] = int(void_in_run)
        rec[f"nobs_{side}"] = nobs
    out.append(rec)

with open(os.environ.get("TRIM_OUT", f"{ROOT}/sim/inputs/trimline_profile.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

hL = np.array([r["h_L_m"] for r in out]); hR = np.array([r["h_R_m"] for r in out])
okL = np.isfinite(hL) & (hL > 0.5); okR = np.isfinite(hR) & (hR > 0.5)
print(f"stations: {len(out)}; with trimline: L={okL.sum()} R={okR.sum()}")
both = np.concatenate([hL[okL], hR[okR]])
if len(both):
    print(f"height stats (m): median {np.median(both):.1f}, p90 {np.percentile(both,90):.1f}, max {both.max():.1f}")
    # top reaches
    chs = np.array([r["chainage_m"] for r in out])
    hmax = np.nanmax(np.column_stack([np.where(okL, hL, np.nan), np.where(okR, hR, np.nan)]), axis=1)
    order = np.argsort(-np.nan_to_num(hmax, nan=-1))[:8]
    for j in order:
        if np.isfinite(hmax[j]) and hmax[j] > 0:
            print(f"  km {chs[j]/1000:6.1f}: height {hmax[j]:5.1f} m "
                  f"(L={hL[j]:.1f} R={hR[j]:.1f} voidL={out[j]['void_L']} voidR={out[j]['void_R']})")
