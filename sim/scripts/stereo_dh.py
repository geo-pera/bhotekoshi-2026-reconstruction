"""Elevation change from opposite-look WV03 ortho parallax.

Both orthos were terrain-corrected with a pre-event base DEM; where the flood
changed the surface, they disagree by dh * (tan th1 + tan th2) along the look
azimuth (~NNE). Phase-correlate chips, project the offset, calibrate bias on
stable off-corridor chips.
"""
import csv
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
A_PATH = f"{ROOT}/vantor/B040001100881410.tif"   # az 14.3, off-nadir 21.72
B_PATH = f"{ROOT}/vantor/B040001100881710.tif"   # az 190.5, off-nadir 26.56
TANSUM = np.tan(np.radians(21.72)) + np.tan(np.radians(26.56))
AZ = np.radians(14.3)          # parallax axis (NNE)
CHIP = 512                     # px at working res
WRES = 0.8                     # m, working resolution

dsA = gdal.Open(A_PATH); dsB = gdal.Open(B_PATH)
def grab(ds, lon, lat, half_m):
    gt = ds.GetGeoTransform()
    px = (lon - gt[0]) / gt[1]; py = (lat - gt[3]) / gt[5]
    half_px_x = half_m / (gt[1] * 111320 * np.cos(np.radians(lat)))
    half_px_y = half_m / (-gt[5] * 110540)
    x0 = int(px - half_px_x); y0 = int(py - half_px_y)
    nx = int(2 * half_px_x); ny = int(2 * half_px_y)
    if x0 < 0 or y0 < 0 or x0 + nx > ds.RasterXSize or y0 + ny > ds.RasterYSize:
        return None
    g = ds.GetRasterBand(2).ReadAsArray(x0, y0, nx, ny,
                                        buf_xsize=CHIP, buf_ysize=CHIP)
    return None if g is None else g.astype(np.float64)

def phasecorr(a, b):
    a = a - a.mean(); b = b - b.mean()
    w = np.outer(np.hanning(CHIP), np.hanning(CHIP))
    F = np.fft.fft2(a * w) * np.conj(np.fft.fft2(b * w))
    F /= np.abs(F) + 1e-12
    corr = np.abs(np.fft.ifft2(F))
    p = np.unravel_index(np.argmax(corr), corr.shape)
    peak = corr.max()
    # subpixel parabola
    def sub(i, axis_len, cm1, c0, cp1):
        den = cm1 - 2 * c0 + cp1
        return (0.5 * (cm1 - cp1) / den) if abs(den) > 1e-12 else 0.0
    r, c = p
    dr = sub(r, CHIP, corr[(r-1) % CHIP, c], corr[r, c], corr[(r+1) % CHIP, c])
    dc = sub(c, CHIP, corr[r, (c-1) % CHIP], corr[r, c], corr[r, (c+1) % CHIP])
    sr = (r if r < CHIP // 2 else r - CHIP) + dr
    sc = (c if c < CHIP // 2 else c - CHIP) + dc
    return sr, sc, peak

cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
ch = np.array([float(r["chainage_m"]) for r in cl])
lons = np.array([float(r["lon"]) for r in cl]); lats = np.array([float(r["lat"]) for r in cl])
xs = np.array([float(r["x_utm45"]) for r in cl]); ys = np.array([float(r["y_utm45"]) for r in cl])

HALF_M = CHIP * WRES / 2
def measure(lon, lat):
    a = grab(dsA, lon, lat, HALF_M); b = grab(dsB, lon, lat, HALF_M)
    if a is None or b is None: return None
    # reject cloud (bright + flat) and nodata
    for g in (a, b):
        if g.mean() > 215 or g.std() < 6 or (g == 0).mean() > 0.2: return None
    sr, sc, peak = phasecorr(a, b)
    if peak < 0.02 or abs(sr) > 200 or abs(sc) > 200: return None
    # offset of B relative to A in metres (x east, y north)
    offx = sc * WRES; offy = -sr * WRES
    # project onto parallax axis (azimuth AZ from north)
    along = offx * np.sin(AZ) + offy * np.cos(AZ)
    return along, peak

# stable bias chips: perpendicular offsets +-(700,1000) m from the channel
bias = []
res_rows = []
for s in np.arange(26000, 39600, 300.0):
    i = int(np.clip(np.searchsorted(ch, s), 5, len(ch) - 6))
    lon = np.interp(s, ch, lons); lat = np.interp(s, ch, lats)
    dx = xs[i+5] - xs[i-5]; dy = ys[i+5] - ys[i-5]; L = np.hypot(dx, dy) or 1.0
    nxv, nyv = -dy / L, dx / L
    m = measure(lon, lat)
    if m: res_rows.append((s, m[0], m[1]))
    for off in (-900.0, 900.0):
        lon2 = lon + nxv * off / (111320 * np.cos(np.radians(lat)))
        lat2 = lat + nyv * off / 110540
        mb = measure(lon2, lat2)
        if mb: bias.append(mb[0])

bias = np.array(bias)
b0 = np.median(bias) if len(bias) else 0.0
print(f"bias chips: {len(bias)}, median along-axis offset {b0:+.2f} m (spread {np.std(bias):.2f})")
print(f"corridor chips: {len(res_rows)}")
print(f"{'km':>6} {'dh(m)':>7} {'peak':>6}")
out = []
for s, along, peak in res_rows:
    dh = (along - b0) / TANSUM
    out.append((s, dh, peak))
    if abs(dh) > 2:
        print(f"{s/1000:6.1f} {dh:+7.1f} {peak:6.3f}")
with open(f"{ROOT}/sim/inputs/stereo_dh.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["chainage_m", "dh_m", "peak"])
    for s, dh, peak in out: w.writerow([f"{s:.0f}", f"{dh:.2f}", f"{peak:.3f}"])
dhv = np.array([o[1] for o in out])
if len(dhv):
    print(f"\nsummary: median {np.median(dhv):+.1f} m, p10 {np.percentile(dhv,10):+.1f}, "
          f"p90 {np.percentile(dhv,90):+.1f}")
print("wrote sim/inputs/stereo_dh.csv")
