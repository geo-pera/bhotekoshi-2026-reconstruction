"""Flood/debris-flow disturbance mapping on the corridor 8 m grid.

Per scene: warp analytic bands + OCM cloud mask onto the corridor grid, compute
NDVI + brightness, classify vegetated (per-scene Otsu, clamped) and bright.
Composite: disturbance = vegetated pre-event AND (non-vegetated AND bright) post-event,
observed clear in both. Then keep only components connected to the river channel.

Outputs (sim/masks/):
  scenebits_<id>.tif   uint8 bitfield: 1=clear-valid 2=veg 4=bright 8=water
  disturbance.tif      uint8: 1=disturbed(channel-connected) 2=disturbed(detached)
                             0=observed not disturbed, 255=not observed clear in both
  disturbance_quicklook.png
"""
import glob, sys
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
GRID = f"{ROOT}/sim/dem/domain_8m_filled_ortho.tif"
g = gdal.Open(GRID)
GT, PRJ, NX, NY = g.GetGeoTransform(), g.GetProjection(), g.RasterXSize, g.RasterYSize
BOUNDS = (GT[0], GT[3] + GT[5] * NY, GT[0] + GT[1] * NX, GT[3])

import os
_ids = sorted(os.path.basename(f)[:-10] for f in glob.glob(f"{ROOT}/sim/masks/*_cloud.tif"))

PRE = [i for i in _ids if i.startswith("202605")]
POST = [i for i in _ids if i.startswith("202608")]
print(f"scenes discovered: {len(PRE)} pre, {len(POST)} post", flush=True)

def warp_band(path, band, resamp="bilinear"):
    w = gdal.Warp('', path, format='MEM', dstSRS=PRJ, outputBounds=BOUNDS,
                  xRes=8, yRes=8, resampleAlg=resamp)
    a = w.GetRasterBand(band).ReadAsArray()
    return a

def otsu(vals, lo, hi):
    h, edges = np.histogram(vals, bins=256, range=(-0.2, 0.9))
    h = h.astype(np.float64); total = h.sum()
    if total == 0: return (lo + hi) / 2
    w1 = np.cumsum(h); mu = np.cumsum(h * edges[:-1]); muT = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        sb = (muT * w1 - mu) ** 2 / (w1 * (total - w1))
    t = edges[np.nanargmax(sb)]
    return float(np.clip(t, lo, hi))

def scene_bits(sid, is_post):
    apath = glob.glob(f"{ROOT}/nepal-flash-flood-2026-08-26/*/*/items/{sid}/{sid}_analytic*.tif")[0]
    cpath = f"{ROOT}/sim/masks/{sid}_cloud.tif"
    wpath = f"{ROOT}/sim/masks/{sid}_water.tif"
    r = warp_band(apath, 3).astype(np.float32)
    n = warp_band(apath, 4).astype(np.float32)
    gband = warp_band(apath, 2).astype(np.float32)
    b = warp_band(apath, 1).astype(np.float32)
    cloud = warp_band(cpath, 1, "nearest")
    if os.path.exists(wpath):
        water = warp_band(wpath, 1, "nearest")
    else:
        water = np.zeros_like(cloud)
    valid = (r + n + gband + b) > 0
    clear = valid & (cloud == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(valid, (n - r) / (n + r + 1e-6), np.nan)
    bright = (r + gband + b) / 3.0
    lo, hi = ((0.05, 0.45) if is_post else (0.20, 0.60))  # radiance vs SR clamps
    t = otsu(ndvi[clear & np.isfinite(ndvi)], lo, hi)
    landclear = clear & (water != 1)
    bt = np.nanmedian(bright[landclear]) if landclear.any() else np.inf
    bits = np.zeros((NY, NX), np.uint8)
    bits |= clear.astype(np.uint8)
    bits |= ((ndvi > t) & clear).astype(np.uint8) << 1
    bits |= ((bright > bt) & clear).astype(np.uint8) << 2
    bits |= (water == 1).astype(np.uint8) << 3
    drv = gdal.GetDriverByName('GTiff')
    out = drv.Create(f"{ROOT}/sim/masks/scenebits_{sid}.tif", NX, NY, 1, gdal.GDT_Byte,
                     ['COMPRESS=DEFLATE', 'TILED=YES'])
    out.SetGeoTransform(GT); out.SetProjection(PRJ)
    out.GetRasterBand(1).WriteArray(bits); out.FlushCache(); out = None
    print(f"{sid}: ndvi_thresh={t:.3f} clear={clear.mean()*100:.1f}% of grid "
          f"veg|clear={(bits&2>0).sum()/max(clear.sum(),1)*100:.0f}%", flush=True)
    return bits

pre_clear = np.zeros((NY, NX), bool); pre_veg = np.zeros((NY, NX), bool)
for sid in PRE:
    bits = scene_bits(sid, False)
    c = (bits & 1) > 0
    pre_veg |= ((bits & 2) > 0)
    pre_clear |= c
post_clear = np.zeros((NY, NX), bool)
post_veg = np.zeros((NY, NX), bool); post_nonvegbright = np.zeros((NY, NX), bool)
post_water = np.zeros((NY, NX), bool)
for sid in POST:
    bits = scene_bits(sid, True)
    c = (bits & 1) > 0
    post_clear |= c
    post_veg |= ((bits & 2) > 0)
    post_nonvegbright |= (c & ((bits & 2) == 0) & ((bits & 4) > 0))
    post_water |= ((bits & 8) > 0)

obs = pre_clear & post_clear
# disturbed: was vegetated, now (bare & bright) or now water, and never seen vegetated post-event
dist = obs & pre_veg & (~post_veg) & (post_nonvegbright | post_water)

# channel connectivity via BFS from centerline buffer
import csv, collections
seed = np.zeros((NY, NX), bool)
for row in csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")):
    x, y = float(row["x_utm45"]), float(row["y_utm45"])
    c = int((x - GT[0]) / GT[1]); r = int((y - GT[3]) / GT[5])
    if 0 <= r < NY and 0 <= c < NX:
        seed[max(0, r-13):r+13, max(0, c-13):c+13] = True   # ~100 m buffer
# label disturbance, keep components touching seed (iterative dilation-limited BFS)
try:
    from scipy import ndimage as ndi
    lab, nlab = ndi.label(dist, structure=np.ones((3, 3)))
    keep = np.unique(lab[seed & (lab > 0)])
    connected = np.isin(lab, keep[keep > 0])
except Exception:
    # numpy-only BFS flood fill from seeds
    connected = np.zeros_like(dist)
    frontier = dist & seed
    while frontier.any():
        connected |= frontier
        grow = np.zeros_like(dist)
        grow[1:, :] |= frontier[:-1, :]; grow[:-1, :] |= frontier[1:, :]
        grow[:, 1:] |= frontier[:, :-1]; grow[:, :-1] |= frontier[:, 1:]
        frontier = grow & dist & ~connected
out = np.full((NY, NX), 255, np.uint8)
out[obs] = 0
out[dist] = 2
out[connected] = 1
drv = gdal.GetDriverByName('GTiff')
o = drv.Create(f"{ROOT}/sim/masks/disturbance.tif", NX, NY, 1, gdal.GDT_Byte,
               ['COMPRESS=DEFLATE', 'TILED=YES'])
o.SetGeoTransform(GT); o.SetProjection(PRJ)
ob = o.GetRasterBand(1); ob.WriteArray(out); ob.SetNoDataValue(255); o.FlushCache(); o = None
print(f"observed both: {obs.mean()*100:.1f}% of grid; disturbed: {dist.sum()*64/1e6:.2f} km^2; "
      f"channel-connected: {connected.sum()*64/1e6:.2f} km^2", flush=True)

# quicklook
from PIL import Image
sub = out[::4, ::4]
ql = np.zeros(sub.shape + (3,), np.uint8)
ql[sub == 255] = [40, 40, 40]
ql[sub == 0] = [90, 120, 90]
ql[sub == 2] = [200, 160, 40]
ql[sub == 1] = [220, 40, 40]
Image.fromarray(ql).save(f"{ROOT}/sim/masks/disturbance_quicklook.png")
print("wrote disturbance.tif + quicklook", flush=True)
