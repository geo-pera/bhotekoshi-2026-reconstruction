"""Viewer bundle v2: PNG-packed heightmap + water sprite sheet + 2048px texture."""
import sys, json, base64, io, os
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
from PIL import Image

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
NPZ = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/sim/runs/swe2d_gorge_12m_postDEM.npz"
WSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TXW = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
d = np.load(NPZ)
Z = d["Z"].astype(np.float32); snaps = d["snaps"]; st = d["snap_t"]; gt = d["gt"]
NY, NX = Z.shape
print(f"terrain {NX}x{NY} @ {gt[1]} m, {len(snaps)} frames")

def png_b64(img_arr, mode):
    buf = io.BytesIO(); Image.fromarray(img_arr, mode).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

# terrain: decimetres, 16-bit split into R (hi) G (lo)
zmin = float(Z.min())
Zu = ((Z - zmin) * 10).clip(0, 65535).astype(np.uint16)
hgt = np.zeros((NY, NX, 3), np.uint8)
hgt[..., 0] = Zu >> 8; hgt[..., 1] = Zu & 255
terr_png = png_b64(hgt, "RGB")
print(f"heightmap PNG: {len(terr_png)/1e6:.2f} MB b64")

# water sprite sheet: all frames, 24 m, 0.5 m steps
frames = [ (snaps[i][::WSTEP, ::WSTEP] * 2).clip(0, 255).astype(np.uint8) for i in range(len(snaps)) ]
wNY, wNX = frames[0].shape
NF = len(frames)
COLS = 8; ROWS = (NF + COLS - 1) // COLS
sheet = np.zeros((ROWS * wNY, COLS * wNX), np.uint8)
for k, f in enumerate(frames):
    r, c = divmod(k, COLS)
    sheet[r*wNY:(r+1)*wNY, c*wNX:(c+1)*wNX] = f
water_png = png_b64(sheet, "L")
print(f"water sheet {COLS}x{ROWS} of {wNX}x{wNY}: {len(water_png)/1e6:.2f} MB b64")

# texture 2048 px
x0, y1 = gt[0], gt[3]; x1 = x0 + gt[1]*NX; y0 = y1 + gt[5]*NY
TX = TXW; TY = int(TX * NY / NX)
w = gdal.Warp('', f"{ROOT}/vantor/10300100C86CED00.tif", format='MEM', dstSRS="EPSG:32645",
              outputBounds=(x0, y0, x1, y1), width=TX, height=TY, resampleAlg='cubic')
rgb = np.stack([w.GetRasterBand(b).ReadAsArray() for b in (1, 2, 3)], -1)
gy, gx = np.gradient(Z, gt[1])
slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
hs = np.clip(np.sin(np.radians(45))*np.cos(slope) +
             np.cos(np.radians(45))*np.sin(slope)*np.cos(np.radians(315)-aspect), 0, 1)
hs_img = np.array(Image.fromarray((hs*200+30).astype(np.uint8)).resize((TX, TY)))
mask = rgb.sum(-1) == 0
for c in range(3): rgb[..., c][mask] = hs_img[mask]
buf = io.BytesIO(); Image.fromarray(rgb.astype(np.uint8)).save(buf, format="JPEG", quality=78)
tex = base64.b64encode(buf.getvalue()).decode()
print(f"texture {TX}x{TY}: {len(tex)/1e6:.2f} MB b64")

out = dict(nx=NX, ny=NY, res=float(gt[1]), zmin=zmin,
           wnx=wNX, wny=wNY, wres=float(gt[1]*WSTEP), cols=COLS,
           times=[float(t) for t in st],
           terrain_png=terr_png, water_png=water_png, texture=tex)
path = f"{ROOT}/sim/runs/viewer_bundle3.json"
json.dump(out, open(path, "w"))
print(f"bundle2: {os.path.getsize(path)/1e6:.1f} MB")
