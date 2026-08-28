"""GPU (torch/MPS) port of the Audusse 2D SWE engine — breach scenario CLI.
Usage: swe2d_torch.py [VOL_Mm3] [T_END_min] [RES] [OUT]"""
import sys, os
import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
import torch

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
VOL = (float(sys.argv[1]) if len(sys.argv) > 1 else 5.0) * 1e6
T_END = (float(sys.argv[2]) if len(sys.argv) > 2 else 360.0) * 60.0
RES = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
OUT = sys.argv[4] if len(sys.argv) > 4 else f"{ROOT}/sim/runs/swe2d_lende_breach_gpu.npz"
G = 9.81; NMAN = 0.06; HMIN = 0.02
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {dev}", flush=True)

xs_, ys_ = rio_transform("EPSG:4326", "EPSG:32645", [85.30, 85.56], [28.21, 28.35])
bx0, bx1 = min(xs_), max(xs_); by0, by1 = min(ys_), max(ys_)
with rasterio.open(f"{ROOT}/sim/dem/domain_8m_postevent.tif") as src:
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    NXg = int((bx1 - bx0) / RES); NYg = int((by1 - by0) / RES)
    dstt = from_origin(bx0, by1, RES, RES)
    Znp = np.zeros((NYg, NXg), np.float32)
    reproject(rasterio.band(src, 1), Znp, dst_transform=dstt, dst_crs="EPSG:32645",
              resampling=Resampling.bilinear)
gt = (bx0, RES, 0, by1, 0, -RES)
print(f"grid {NXg}x{NYg} at {RES} m; V={VOL/1e6:.0f} Mm3", flush=True)

# stream-burn the conditioned channel profile (removes DEM sink artifacts)
BURN = os.environ.get("BURN")
if BURN:
    prof = np.load(BURN)   # rows: dist, x_utm, y_utm, z_conditioned
    n_burn = 0
    R1, R2 = 40.0, 90.0
    for dd, px, py, pz in prof:
        ci = int((gt[3] - py) / RES); cj = int((px - gt[0]) / RES)
        r2 = int(R2 / RES) + 1
        for ii in range(max(0, ci - r2), min(NYg, ci + r2 + 1)):
            for jj in range(max(0, cj - r2), min(NXg, cj + r2 + 1)):
                dist = np.hypot((ii - ci) * RES, (jj - cj) * RES)
                if dist <= R1:
                    tgtz = pz
                elif dist <= R2:
                    tgtz = pz + (dist - R1) / (R2 - R1) * 20.0
                else:
                    continue
                if Znp[ii, jj] > tgtz:
                    Znp[ii, jj] = tgtz; n_burn += 1
    print(f"burned channel: {n_burn} cells lowered", flush=True)

Z = torch.tensor(Znp, device=dev)
tp_, kk = 1200.0, 3.0
tt = np.arange(0, 4 * 3600, 5.0)
raw = (tt / tp_ * np.exp(1 - tt / tp_)) ** kk
scale = VOL / np.trapezoid(raw, tt)
def qin_t(t): return scale * ((t / tp_ * np.exp(1 - min(t / tp_, 50))) ** kk) if t > 0 else 0.0

lx, ly = rio_transform("EPSG:4326", "EPSG:32645", [85.525], [28.318])
li = int((gt[3] - ly[0]) / RES); lj = int((lx[0] - gt[0]) / RES)
srcm = np.zeros((NYg, NXg), np.float32)
rad = int(120 / RES) + 1
for dj in range(0, int(500 / RES)):
    srcm[max(0, li - rad):li + rad + 1, max(0, lj - dj - rad):lj - dj + rad + 1] = 1
nsrc = srcm.sum()
srcT = torch.tensor(srcm / nsrc / (RES * RES), device=dev)   # per-unit-Q depth increment
dirx, diry = -0.99, -0.10; U_INJ = 5.0
print(f"injection cells: {int(nsrc)}", flush=True)

sp = np.ones((NYg, NXg), np.float32); SP = 10
ramp = np.linspace(0.85, 1.0, SP).astype(np.float32)
sp[:SP, :] *= ramp[:, None]; sp[-SP:, :] *= ramp[::-1][:, None]
sp[:, :SP] *= ramp[None, :]; sp[:, -SP:] *= ramp[::-1][None, :]
sponge = torch.tensor(sp, device=dev)

h = torch.zeros_like(Z); qx = torch.zeros_like(Z); qy = torch.zeros_like(Z)
peak_eta = torch.full_like(Z, -1e30); peak_depth = torch.zeros_like(Z)
snaps = []; snap_t = []; next_snap = 0.0

def vel(h, q): return torch.where(h > HMIN, q / torch.clamp(h, min=HMIN), torch.zeros_like(q))
def sweep(h, qn, qt, axis):
    un = torch.clamp(vel(h, qn), -40, 40); ut = torch.clamp(vel(h, qt), -40, 40)
    hR = torch.roll(h, -1, axis); unR = torch.roll(un, -1, axis); utR = torch.roll(ut, -1, axis)
    zR = torch.roll(Z, -1, axis); zI = torch.maximum(Z, zR)
    hLs = torch.clamp(h + Z - zI, min=0.0); hRs = torch.clamp(hR + zR - zI, min=0.0)
    qLs = hLs * un; qRs = hRs * unR
    a = torch.maximum(un.abs() + torch.sqrt(G * hLs), unR.abs() + torch.sqrt(G * hRs))
    Fh = 0.5 * (qLs + qRs) - 0.5 * a * (hRs - hLs)
    Fq = 0.5 * (qLs * un + 0.5 * G * hLs ** 2 + qRs * unR + 0.5 * G * hRs ** 2) - 0.5 * a * (qRs - qLs)
    Ft = torch.where(Fh >= 0, Fh * ut, Fh * utR)
    dh = Fh - torch.roll(Fh, 1, axis)
    dqn = Fq - torch.roll(Fq, 1, axis) - 0.5 * G * (hLs ** 2 - torch.roll(hRs, 1, axis) ** 2)
    dqt = Ft - torch.roll(Ft, 1, axis)
    return dh, dqn, dqt

t = 0.0; nstep = 0; dt = 0.5
with torch.no_grad():
    while t < T_END:
        if nstep % 10 == 0:
            spd = torch.hypot(vel(h, qx), vel(h, qy))
            cmax = float((spd + torch.sqrt(G * torch.clamp(h, min=0))).max())
            dt = min(0.22 * RES / max(cmax, 0.8), 2.0)
        u = torch.clamp(vel(h, qx), -40, 40); v = torch.clamp(vel(h, qy), -40, 40)
        qx = u * h; qy = v * h
        dhx, dqxx, dqyx = sweep(h, qx, qy, 1)
        dhy, dqyy, dqxy = sweep(h, qy, qx, 0)
        hn = h - dt / RES * (dhx + dhy)
        qxn = qx - dt / RES * (dqxx + dqxy)
        qyn = qy - dt / RES * (dqyx + dqyy)
        spd2 = torch.hypot(u, v)
        fr = G * NMAN ** 2 * spd2 / torch.clamp(h, min=HMIN) ** (4.0 / 3.0)
        qxn = qxn / (1 + dt * fr); qyn = qyn / (1 + dt * fr)
        qv = qin_t(t)
        add = srcT * (dt * qv)
        hn = hn + add
        qxn = qxn + add * (U_INJ * dirx); qyn = qyn + add * (U_INJ * (-diry))
        hn = hn * sponge; qxn = qxn * sponge; qyn = qyn * sponge
        h = torch.clamp(hn, min=0.0)
        wet = h > HMIN
        qx = torch.where(wet, qxn, torch.zeros_like(qxn))
        qy = torch.where(wet, qyn, torch.zeros_like(qyn))
        peak_eta = torch.where(wet, torch.maximum(peak_eta, Z + h), peak_eta)
        peak_depth = torch.maximum(peak_depth, h)
        if t >= next_snap:
            snaps.append(h.cpu().numpy().astype(np.float32)); snap_t.append(t); next_snap += 120.0
        if nstep % 4000 == 0:
            mx = float(h.max())
            print(f"  t={t/60:6.1f} min dt={dt:.2f}s maxdepth={mx:6.1f}", flush=True)
            if not np.isfinite(mx) or mx > 300:
                print("UNSTABLE"); break
        t += dt; nstep += 1

np.savez_compressed(OUT, peak_eta=peak_eta.cpu().numpy(), peak_depth=peak_depth.cpu().numpy(),
                    Z=Znp, gt=np.array(gt), snaps=np.array(snaps), snap_t=np.array(snap_t))
print(f"done: {nstep} steps, {len(snaps)} snapshots -> {OUT}", flush=True)
