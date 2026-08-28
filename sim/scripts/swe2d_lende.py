"""2D breach-scenario model: barrier lake (Chhochen-Purepu confluence) down the
Lende Khola to Timure/Rasuwagadhi. Same Audusse scheme as swe2d.py.
Usage: swe2d_lende.py [VOL_Mm3] [T_END_min] [RES]"""
import sys, csv
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
VOL = (float(sys.argv[1]) if len(sys.argv) > 1 else 5.0) * 1e6
T_END = (float(sys.argv[2]) if len(sys.argv) > 2 else 360.0) * 60.0
RES = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
G = 9.81; NMAN = 0.06; HMIN = 0.02

sr = osr.SpatialReference(); sr.ImportFromEPSG(4326); sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
srU = osr.SpatialReference(); srU.ImportFromEPSG(32645); srU.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
tr = osr.CoordinateTransformation(sr, srU)
x0, y0, _ = tr.TransformPoint(85.30, 28.21); x1, y1, _ = tr.TransformPoint(85.56, 28.35)
w = gdal.Warp('', f"{ROOT}/sim/dem/domain_8m_postevent.tif", format='MEM',
              outputBounds=(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)),
              xRes=RES, yRes=RES, resampleAlg='bilinear')
Z = w.GetRasterBand(1).ReadAsArray().astype(np.float64)
gt = w.GetGeoTransform(); NYg, NXg = Z.shape
print(f"grid {NXg}x{NYg} at {RES} m; V={VOL/1e6:.0f} Mm3", flush=True)

def rc(lon, lat):
    x, y, _ = tr.TransformPoint(lon, lat)
    return int((y - gt[3]) / gt[5]), int((x - gt[0]) / gt[1])

# breach hydrograph: erosional gamma, peak at ~20 min
tp, kk = 1200.0, 3.0
tt = np.arange(0, 4 * 3600, 5.0)
raw = (tt / tp * np.exp(1 - tt / tp)) ** kk
scale = VOL / np.trapezoid(raw, tt)
def qin_t(t): return scale * ((t / tp * np.exp(1 - min(t / tp, 50))) ** kk) if t > 0 else 0.0
print(f"peak breach outflow ~{scale * (1.0)**kk:.0f} m3/s", flush=True)

# injection strip at the lake site, flowing west
li, lj = rc(85.525, 28.318)
src = np.zeros((NYg, NXg), bool)
rad = int(120 / RES) + 1
for dj in range(0, int(500 / RES)):
    src[max(0, li - rad):li + rad + 1, max(0, lj - dj - rad):lj - dj + rad + 1] = True
nsrc = src.sum()
dirx, diry = -0.99, -0.10
U_INJ = 5.0
print(f"injection {nsrc} cells at lake site", flush=True)

sponge = np.ones((NYg, NXg)); SP = 10
ramp = np.linspace(0.85, 1.0, SP)
sponge[:SP, :] *= ramp[:, None]; sponge[-SP:, :] *= ramp[::-1][:, None]
sponge[:, :SP] *= ramp[None, :]; sponge[:, -SP:] *= ramp[::-1][None, :]

h = np.zeros_like(Z); qx = np.zeros_like(Z); qy = np.zeros_like(Z)
peak_eta = np.full_like(Z, -np.inf); peak_depth = np.zeros_like(Z)
snaps = []; snap_t = []; next_snap = 0.0
def vel(h, q): return np.where(h > HMIN, q / np.maximum(h, HMIN), 0.0)
def sweep(h, qn, qt, Z, axis):
    un = np.clip(vel(h, qn), -40, 40); ut = np.clip(vel(h, qt), -40, 40)
    hR = np.roll(h, -1, axis); unR = np.roll(un, -1, axis); utR = np.roll(ut, -1, axis)
    zL = Z; zR = np.roll(Z, -1, axis); zI = np.maximum(zL, zR)
    hLs = np.clip(h + zL - zI, 0.0, None); hRs = np.clip(hR + zR - zI, 0.0, None)
    qLs = hLs * un; qRs = hRs * unR
    a = np.maximum(np.abs(un) + np.sqrt(G * hLs), np.abs(unR) + np.sqrt(G * hRs))
    Fh = 0.5 * (qLs + qRs) - 0.5 * a * (hRs - hLs)
    Fq = 0.5 * (qLs * un + 0.5 * G * hLs ** 2 + qRs * unR + 0.5 * G * hRs ** 2) - 0.5 * a * (qRs - qLs)
    Ft = np.where(Fh >= 0, Fh * ut, Fh * utR)
    dh = Fh - np.roll(Fh, 1, axis)
    dqn = Fq - np.roll(Fq, 1, axis) - 0.5 * G * (hLs ** 2 - np.roll(hRs, 1, axis) ** 2)
    dqt = Ft - np.roll(Ft, 1, axis)
    return dh, dqn, dqt

t = 0.0; nstep = 0
while t < T_END:
    u = np.clip(vel(h, qx), -40, 40); v = np.clip(vel(h, qy), -40, 40)
    qx = u * h; qy = v * h
    spd = np.hypot(u, v)
    cmax = (spd + np.sqrt(G * np.maximum(h, 0))).max()
    dt = min(0.25 * RES / max(cmax, 0.8), 2.0)
    dhx, dqxx, dqyx = sweep(h, qx, qy, Z, 1)
    dhy, dqyy, dqxy = sweep(h, qy, qx, Z, 0)
    hn = h - dt / RES * (dhx + dhy)
    qxn = qx - dt / RES * (dqxx + dqxy)
    qyn = qy - dt / RES * (dqyx + dqyy)
    fr = G * NMAN ** 2 * spd / np.maximum(h, HMIN) ** (4.0 / 3.0)
    qxn /= (1 + dt * fr); qyn /= (1 + dt * fr)
    qv = qin_t(t)
    dh_in = dt * qv / (nsrc * RES * RES)
    hn[src] += dh_in
    qxn[src] += dh_in * U_INJ * dirx; qyn[src] += dh_in * U_INJ * (-diry)
    hn *= sponge; qxn *= sponge; qyn *= sponge
    h = np.maximum(hn, 0.0)
    qx = np.where(h > HMIN, qxn, 0.0); qy = np.where(h > HMIN, qyn, 0.0)
    wet = h > HMIN
    peak_eta = np.where(wet, np.maximum(peak_eta, Z + h), peak_eta)
    peak_depth = np.maximum(peak_depth, h)
    if t >= next_snap:
        snaps.append(h.astype(np.float32)); snap_t.append(t); next_snap += 120.0
    if nstep % 4000 == 0:
        print(f"  t={t/60:6.1f} min dt={dt:.2f}s wet={wet.sum()*RES*RES/1e6:5.2f} km2 "
              f"maxdepth={h.max():6.1f} maxspd={spd.max():5.1f}", flush=True)
    if not np.isfinite(h).all() or h.max() > 300:
        print(f"UNSTABLE t={t:.0f}"); break
    t += dt; nstep += 1

np.savez_compressed(f"{ROOT}/sim/runs/swe2d_lende_breach.npz",
                    peak_eta=peak_eta, peak_depth=peak_depth, Z=Z.astype(np.float32),
                    gt=np.array(gt), snaps=np.array(snaps), snap_t=np.array(snap_t))
print(f"done: {nstep} steps, {len(snaps)} snapshots", flush=True)
