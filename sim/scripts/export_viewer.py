"""Export 3D viewer bundle: terrain u16, draped-imagery texture JPEG,
water-depth frames u8, as base64 strings in a JSON file."""
import sys, json, base64, io
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
from PIL import Image

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
NPZ = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/sim/runs/swe2d_gorge_24m.npz"
WSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2     # water downsample factor
FSTEP = int(sys.argv[3]) if len(sys.argv) > 3 else 2     # frame subsample

d = np.load(NPZ)
Z = d["Z"].astype(np.float32); snaps = d["snaps"]; st = d["snap_t"]; gt = d["gt"]
NY, NX = Z.shape
print(f"grid {NX}x{NY} at {gt[1]} m, {len(snaps)} frames")

# terrain as u16 (relative to zmin, cm... use dm to fit)
zmin = float(Z.min())
Zu = ((Z - zmin) * 10).clip(0, 65535).astype(np.uint16)   # decimetres
terr_b64 = base64.b64encode(Zu.tobytes()).decode()

# water frames: downsample, u8 in 0.5 m steps (max 127 m)
frames = []
times = []
for i in range(0, len(snaps), FSTEP):
    w = snaps[i][::WSTEP, ::WSTEP]
    frames.append((w * 2).clip(0, 255).astype(np.uint8).tobytes())
    times.append(float(st[i]))
wNY, wNX = snaps[0][::WSTEP, ::WSTEP].shape
water_b64 = base64.b64encode(b"".join(frames)).decode()
print(f"water {wNX}x{wNY} x {len(frames)} frames = {len(water_b64)/1e6:.1f} MB b64")

# texture: WV02-2021 draped, gaps -> hillshade grey
x0, y1 = gt[0], gt[3]; x1 = x0 + gt[1]*NX; y0 = y1 + gt[5]*NY
TX = 1024; TY = int(TX * NY / NX)
w = gdal.Warp('', f"{ROOT}/vantor/10300100C86CED00.tif", format='MEM', dstSRS="EPSG:32645",
              outputBounds=(x0, y0, x1, y1), width=TX, height=TY, resampleAlg='bilinear')
rgb = np.stack([w.GetRasterBand(b).ReadAsArray() for b in (1, 2, 3)], -1)
# hillshade fallback
gy, gx = np.gradient(Z, gt[1])
hs = np.clip(np.sin(np.radians(45))*np.cos(np.arctan(np.hypot(gx, gy))) +
             np.cos(np.radians(45))*np.sin(np.arctan(np.hypot(gx, gy))) *
             np.cos(np.radians(315)-np.arctan2(-gx, gy)), 0, 1)
hs_img = np.array(Image.fromarray((hs*200+30).astype(np.uint8)).resize((TX, TY)))
mask = rgb.sum(-1) == 0
for c in range(3): rgb[..., c][mask] = hs_img[mask]
buf = io.BytesIO()
Image.fromarray(rgb.astype(np.uint8)).save(buf, format="JPEG", quality=80)
tex_b64 = base64.b64encode(buf.getvalue()).decode()
print(f"texture {TX}x{TY} = {len(tex_b64)/1e6:.2f} MB b64; imagery covers {(~mask).mean()*100:.0f}%")

out = dict(nx=NX, ny=NY, res=float(gt[1]), zmin=zmin,
           wnx=wNX, wny=wNY, wres=float(gt[1]*WSTEP),
           times=times, terrain=terr_b64, water=water_b64, texture=tex_b64)
path = f"{ROOT}/sim/runs/viewer_bundle.json"
json.dump(out, open(path, "w"))
import os; print(f"bundle: {os.path.getsize(path)/1e6:.1f} MB")
