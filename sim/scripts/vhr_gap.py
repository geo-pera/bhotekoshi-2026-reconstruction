"""Fill the km 16-34 disturbance gap using VHR day-after imagery.

Pre baseline: WV02 Oct-2021 (RGB -> VARI vegetation).
Post: WV03 strips (RGB -> VARI), SkySat + Pelican (BGRN -> NDVI, udm2-gated).
Disturbance = pre-veg AND post-bare-and-bright, channel-connected, merged into
disturbance_v2.tif (only within the gap zone; elsewhere the validated Planet
product is kept).
"""
import glob, csv
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
g = gdal.Open(f"{ROOT}/sim/dem/domain_8m_filled_ortho.tif")
GT, PRJ, NX, NY = g.GetGeoTransform(), g.GetProjection(), g.RasterXSize, g.RasterYSize
BOUNDS = (GT[0], GT[3] + GT[5] * NY, GT[0] + GT[1] * NX, GT[3])

def warp(path, band, alg="average"):
    w = gdal.Warp('', path, format='MEM', dstSRS=PRJ, outputBounds=BOUNDS,
                  xRes=8, yRes=8, resampleAlg=alg)
    a = w.GetRasterBand(band).ReadAsArray()
    return a.astype(np.float32)

def otsu(vals, lo, hi, rng):
    h, e = np.histogram(vals, bins=256, range=rng)
    h = h.astype(np.float64); tot = h.sum()
    if tot == 0: return (lo + hi) / 2
    w1 = np.cumsum(h); mu = np.cumsum(h * e[:-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        sb = (mu[-1] * w1 - mu) ** 2 / (w1 * (tot - w1))
    return float(np.clip(e[np.nanargmax(sb)], lo, hi))

def rgb_layers(path, name):
    r = warp(path, 1); gg = warp(path, 2); b = warp(path, 3)
    valid = (r + gg + b) > 0
    bright = (r + gg + b) / 3
    white = np.minimum(np.minimum(r, gg), b)
    cloud = valid & (white > 190)
    dark = valid & (bright < 25)
    clear = valid & ~cloud & ~dark
    with np.errstate(divide='ignore', invalid='ignore'):
        vari = np.where(clear, (gg - r) / (gg + r - b + 1e-3), np.nan)
    t = otsu(vari[clear & np.isfinite(vari)], 0.0, 0.12, (-0.4, 0.5))
    veg = clear & (vari > t)
    bt = np.nanmedian(bright[clear])
    brightm = clear & (bright > bt)
    print(f"{name}: clear {clear.mean()*100:.1f}% of grid, vari_t={t:.3f}, "
          f"veg|clear={veg.sum()/max(clear.sum(),1)*100:.0f}%", flush=True)
    return clear, veg, brightm

def bgrn_layers(path, name, udm2=None):
    b = warp(path, 1); gg = warp(path, 2); r = warp(path, 3); n = warp(path, 4)
    valid = (b + gg + r + n) > 0
    if udm2 is not None:
        cl = warp(udm2, 1, "nearest")
        clear = valid & (cl == 1)
    else:
        bright8 = (r + gg + b) / np.nanmax([(r + gg + b).max(), 1]) * 255
        clear = valid & (np.minimum(np.minimum(r, gg), b) < np.nanpercentile(b[valid], 97))
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(clear, (n - r) / (n + r + 1e-3), np.nan)
    t = otsu(ndvi[clear & np.isfinite(ndvi)], 0.15, 0.55, (-0.2, 0.9))
    veg = clear & (ndvi > t)
    bright = (r + gg + b) / 3
    bt = np.nanmedian(bright[clear])
    brightm = clear & (bright > bt)
    print(f"{name}: clear {clear.mean()*100:.1f}% of grid, ndvi_t={t:.3f}, "
          f"veg|clear={veg.sum()/max(clear.sum(),1)*100:.0f}%", flush=True)
    return clear, veg, brightm

pre_clear, pre_veg, _ = rgb_layers(f"{ROOT}/vantor/10300100C86CED00.tif", "WV02-2021 pre")

post_clear = np.zeros((NY, NX), bool)
post_veg = np.zeros((NY, NX), bool)
post_bb = np.zeros((NY, NX), bool)
for p, nm in ((f"{ROOT}/vantor/B040001100882F10.tif", "WV03-2F10"),
              (f"{ROOT}/vantor/B040001100881410.tif", "WV03-1410")):
    c, v, br = rgb_layers(p, nm)
    post_clear |= c; post_veg |= v; post_bb |= (c & ~v & br)
for item in glob.glob(f"{ROOT}/nepal-flash-flood-2026-08-26/post-event/skysat-2026-08-27/items/*"):
    ps = glob.glob(f"{item}/*_pansharpened.tif")
    if not ps: continue
    c, v, br = bgrn_layers(ps[0], "skysat " + item.split("/")[-1][-5:])
    post_clear |= c; post_veg |= v; post_bb |= (c & ~v & br)
for item in glob.glob(f"{ROOT}/nepal-flash-flood-2026-08-26/post-event/pelican-2026-08-27/items/*"):
    ps = glob.glob(f"{item}/*_pansharpened.tif")
    ud = glob.glob(f"{item}/*_udm2.tif")
    if not ps: continue
    c, v, br = bgrn_layers(ps[0], "pelican " + item.split("/")[-1][-5:], None)  # udm2 falsely all-cloud per README
    post_clear |= c; post_veg |= v; post_bb |= (c & ~v & br)

obs = pre_clear & post_clear
dist = obs & pre_veg & ~post_veg & post_bb

# gap zone: within 1200 m of centerline, chainage 16-34 km
cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
gap = np.zeros((NY, NX), bool)
seed = np.zeros((NY, NX), bool)
for row in cl:
    s = float(row["chainage_m"])
    if not (16000 <= s <= 34000): continue
    x, y = float(row["x_utm45"]), float(row["y_utm45"])
    c = int((x - GT[0]) / GT[1]); r = int((y - GT[3]) / GT[5])
    if 0 <= r < NY and 0 <= c < NX:
        gap[max(0, r-150):r+150, max(0, c-150):c+150] = True
        seed[max(0, r-13):r+13, max(0, c-13):c+13] = True
dist &= gap
connected = np.zeros_like(dist)
frontier = dist & seed
while frontier.any():
    connected |= frontier
    grow = np.zeros_like(dist)
    grow[1:, :] |= frontier[:-1, :]; grow[:-1, :] |= frontier[1:, :]
    grow[:, 1:] |= frontier[:, :-1]; grow[:, -1:][:, 0] |= frontier[:, -1]
    grow[:, :-1] |= frontier[:, 1:]
    frontier = grow & dist & ~connected
print(f"gap obs: {(obs & gap).sum()*64/1e6:.1f} km2; disturbed: {dist.sum()*64/1e6:.2f} km2; "
      f"connected: {connected.sum()*64/1e6:.2f} km2", flush=True)

base_ds = gdal.Open(f"{ROOT}/sim/masks/disturbance.tif")
base = base_ds.GetRasterBand(1).ReadAsArray()
v2 = base.copy()
v2[gap & obs & (v2 == 255)] = 0
v2[gap & connected] = 1
v2[gap & dist & ~connected & (v2 != 1)] = 2
drv = gdal.GetDriverByName("GTiff")
o = drv.Create(f"{ROOT}/sim/masks/disturbance_v2.tif", NX, NY, 1, gdal.GDT_Byte,
               ["COMPRESS=DEFLATE", "TILED=YES"])
o.SetGeoTransform(GT); o.SetProjection(PRJ)
ob = o.GetRasterBand(1); ob.WriteArray(v2); ob.SetNoDataValue(255); o.FlushCache()
print("wrote sim/masks/disturbance_v2.tif")
