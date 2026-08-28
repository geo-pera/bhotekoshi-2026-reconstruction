"""Breach-scenario viewer bundle: composite texture (WV02 > Planet pre > hillshade)."""
import sys, json, base64, io, os, glob
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
from PIL import Image

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
d = np.load(sys.argv[1] if len(sys.argv)>1 else f"{ROOT}/sim/runs/swe2d_lende_breach.npz")
Z = d["Z"].astype(np.float32); snaps = d["snaps"]; st = d["snap_t"]; gt = d["gt"]
NY, NX = Z.shape
WSTEP = 2; FSTEP = 2
print(f"terrain {NX}x{NY} @ {gt[1]} m, {len(snaps)} frames")

def png_b64(img_arr, mode):
    buf = io.BytesIO(); Image.fromarray(img_arr, mode).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

zmin = float(Z.min())
Zu = ((Z - zmin) * 10).clip(0, 65535).astype(np.uint16)
hgt = np.zeros((NY, NX, 3), np.uint8)
hgt[..., 0] = Zu >> 8; hgt[..., 1] = Zu & 255
terr_png = png_b64(hgt, "RGB")
print(f"heightmap: {len(terr_png)/1e6:.2f} MB b64")

frames = [(snaps[i][::WSTEP, ::WSTEP] * 8).clip(0, 255).astype(np.uint8) for i in range(0, len(snaps), FSTEP)]
times = [float(st[i]) for i in range(0, len(snaps), FSTEP)]
wNY, wNX = frames[0].shape; NF = len(frames)
COLS = 8; ROWS = (NF + COLS - 1) // COLS
sheet = np.zeros((ROWS * wNY, COLS * wNX), np.uint8)
for k, f in enumerate(frames):
    r, c = divmod(k, COLS)
    sheet[r*wNY:(r+1)*wNY, c*wNX:(c+1)*wNX] = f
water_png = png_b64(sheet, "L")
print(f"water sheet {NF} frames @ {wNX}x{wNY} (0.125 m steps): {len(water_png)/1e6:.2f} MB b64")

x0, y1 = gt[0], gt[3]; x1 = x0 + gt[1]*NX; y0 = y1 + gt[5]*NY
TX = 2048; TY = int(TX * NY / NX)
def warp_rgb(src):
    w = gdal.Warp('', src, format='MEM', dstSRS="EPSG:32645",
                  outputBounds=(x0, y0, x1, y1), width=TX, height=TY, resampleAlg='bilinear')
    return np.stack([w.GetRasterBand(b).ReadAsArray() for b in (1, 2, 3)], -1)
gy, gx = np.gradient(Z, gt[1])
slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
hs = np.clip(np.sin(np.radians(45))*np.cos(slope) +
             np.cos(np.radians(45))*np.sin(slope)*np.cos(np.radians(315)-aspect), 0, 1)
tex_arr = np.repeat(np.array(Image.fromarray((hs*190+35).astype(np.uint8)).resize((TX, TY)))[..., None], 3, -1).astype(np.uint8)
pre_vis = sorted(glob.glob(f"{ROOT}/nepal-flash-flood-2026-08-26/pre-event/*/items/*/*_visual.tif"))
planet = warp_rgb(pre_vis)
pm = planet.sum(-1) > 0
tex_arr[pm] = planet[pm]
wv = warp_rgb(f"{ROOT}/vantor/10300100C86CED00.tif")
wm = wv.sum(-1) > 0
tex_arr[wm] = wv[wm]
print(f"texture: WV02 {wm.mean()*100:.0f}%, Planet {(pm & ~wm).mean()*100:.0f}%, hillshade {(~pm & ~wm).mean()*100:.0f}%")
buf = io.BytesIO(); Image.fromarray(tex_arr).save(buf, format="JPEG", quality=78)
tex = base64.b64encode(buf.getvalue()).decode()
print(f"texture {TX}x{TY}: {len(tex)/1e6:.2f} MB b64")

out = dict(nx=NX, ny=NY, res=float(gt[1]), zmin=zmin,
           wnx=wNX, wny=wNY, wres=float(gt[1]*WSTEP), cols=COLS,
           times=times, terrain_png=terr_png, water_png=water_png, texture=tex,
           depth_scale=8)
json.dump(out, open(f"{ROOT}/sim/runs/viewer_bundle_lende.json", "w"))
print(f"bundle: {os.path.getsize(f'{ROOT}/sim/runs/viewer_bundle_lende.json')/1e6:.1f} MB")
