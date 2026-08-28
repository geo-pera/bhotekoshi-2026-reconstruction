"""2D shallow-water model of the Syabrubesi gorge reach (chainage 33-42 km).

Audusse hydrostatic-reconstruction scheme (well-balanced, positivity-preserving
at wet/dry fronts), Rusanov interface dissipation, semi-implicit Manning
friction. Inflow: mass source disk at the km-33 centerline using Q33 from a
route1d run. Saves peak fields + depth snapshots for animation.

Usage: swe2d.py <route1d run .npz with Q33> [RES] [T_END_min]
"""
import sys, csv
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
RUN = sys.argv[1]
RES = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0
T_END = (float(sys.argv[3]) if len(sys.argv) > 3 else 70.0) * 60.0
CH0, CH1 = 33000.0, 42000.0
G = 9.81
NMAN = 0.05
HMIN = 0.02

r = np.load(RUN)
tq = r["t"]; Q33 = r["Q33"]
print(f"hydrograph: peak {Q33.max():.0f} m3/s at t={tq[np.argmax(Q33)]/60:.0f} min", flush=True)

cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
ch = np.array([float(c["chainage_m"]) for c in cl])
xs = np.array([float(c["x_utm45"]) for c in cl]); ys = np.array([float(c["y_utm45"]) for c in cl])
m = (ch >= CH0 - 500) & (ch <= CH1 + 500)
x0, x1 = xs[m].min() - 1200, xs[m].max() + 1200
y0, y1 = ys[m].min() - 1200, ys[m].max() + 1200
import os
DEM_SRC = os.environ.get("DEM", f"{ROOT}/sim/dem/domain_8m_filled_ortho.tif")
w = gdal.Warp('', DEM_SRC, format='MEM',
              outputBounds=(x0, y0, x1, y1), xRes=RES, yRes=RES, resampleAlg='bilinear')
Z = w.GetRasterBand(1).ReadAsArray().astype(np.float64)
gt = w.GetGeoTransform()
NYg, NXg = Z.shape
print(f"grid {NXg}x{NYg} at {RES} m", flush=True)

def rc(x, y):
    return int((y - gt[3]) / gt[5]), int((x - gt[0]) / gt[1])

# injection strip: channel cells along km 33.0-33.6, with down-valley direction
yy, xx = np.mgrid[0:NYg, 0:NXg]
src = np.zeros((NYg, NXg), bool)
for s_ in np.arange(CH0, CH0 + 600, RES / 2):
    px = np.interp(s_, ch, xs); py = np.interp(s_, ch, ys)
    i_, j_ = rc(px, py)
    rad = int(100 / RES) + 1
    src[max(0, i_ - rad):i_ + rad + 1, max(0, j_ - rad):j_ + rad + 1] = True
nsrc = src.sum()
p0x = np.interp(CH0, ch, xs); p0y = np.interp(CH0, ch, ys)
p1x = np.interp(CH0 + 600, ch, xs); p1y = np.interp(CH0 + 600, ch, ys)
dl = np.hypot(p1x - p0x, p1y - p0y)
dirx, diry = (p1x - p0x) / dl, (p1y - p0y) / dl
U_INJ = 15.0
print(f"injection strip: {nsrc} cells, direction ({dirx:+.2f},{diry:+.2f})")

# absorbing sponge at all edges (damps outgoing waves, prevents boundary blow-up)
SP = 10
sponge = np.ones((NYg, NXg))
ramp = np.linspace(0.85, 1.0, SP)
sponge[:SP, :] *= ramp[:, None]; sponge[-SP:, :] *= ramp[::-1][:, None]
sponge[:, :SP] *= ramp[None, :]; sponge[:, -SP:] *= ramp[::-1][None, :]

h = np.zeros_like(Z); qx = np.zeros_like(Z); qy = np.zeros_like(Z)
peak_eta = np.full_like(Z, -np.inf)
peak_speed = np.zeros_like(Z)
snaps = []; snap_t = []; next_snap = 0.0

def vel(h, q):
    return np.where(h > HMIN, q / np.maximum(h, HMIN), 0.0)

def sweep(h, qn, qt, Z, axis):
    """Audusse fluxes along `axis` for normal momentum qn, transverse qt.
    Returns (dh, dqn, dqt) updates per unit dt/dx."""
    un = np.clip(vel(h, qn), -55, 55)
    ut = np.clip(vel(h, qt), -55, 55)
    hR = np.roll(h, -1, axis); unR = np.roll(un, -1, axis); utR = np.roll(ut, -1, axis)
    zL = Z; zR = np.roll(Z, -1, axis)
    zI = np.maximum(zL, zR)
    hLs = np.clip(h + zL - zI, 0.0, None)
    hRs = np.clip(hR + zR - zI, 0.0, None)
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
    u = np.clip(vel(h, qx), -55, 55); v = np.clip(vel(h, qy), -55, 55)
    qx = u * h; qy = v * h
    spd = np.hypot(u, v)
    cmax = (spd + np.sqrt(G * np.maximum(h, 0))).max()
    dt = min(0.25 * RES / max(cmax, 1.0), 1.0)
    dhx, dqxx, dqyx = sweep(h, qx, qy, Z, 1)
    dhy, dqyy, dqxy = sweep(h, qy, qx, Z, 0)
    hn = h - dt / RES * (dhx + dhy)
    qxn = qx - dt / RES * (dqxx + dqxy)
    qyn = qy - dt / RES * (dqyx + dqyy)
    fr = G * NMAN ** 2 * spd / np.maximum(h, HMIN) ** (4.0 / 3.0)
    qxn /= (1 + dt * fr); qyn /= (1 + dt * fr)
    qin = np.interp(t, tq, Q33)
    dh_in = dt * qin / (nsrc * RES * RES)
    hn[src] += dh_in
    qxn[src] += dh_in * U_INJ * dirx
    qyn[src] += dh_in * U_INJ * (-diry)   # row axis points south (gt[5]<0): +y_utm = -row
    hn *= sponge; qxn *= sponge; qyn *= sponge
    h = np.maximum(hn, 0.0)
    qx = np.where(h > HMIN, qxn, 0.0); qy = np.where(h > HMIN, qyn, 0.0)
    wet = h > HMIN
    eta_now = Z + h
    peak_eta = np.where(wet, np.maximum(peak_eta, eta_now), peak_eta)
    peak_speed = np.where(wet, np.maximum(peak_speed, np.hypot(vel(h, qx), vel(h, qy))), peak_speed)
    if t >= next_snap:
        snaps.append(h.astype(np.float32)); snap_t.append(t); next_snap += 30.0
    if nstep % 2000 == 0:
        print(f"  t={t/60:6.1f} min dt={dt:.2f}s wet={wet.sum()*RES*RES/1e6:5.1f} km2 "
              f"maxdepth={h.max():7.1f} maxspd={spd.max():5.1f}", flush=True)
    if not np.isfinite(h).all() or h.max() > 500:
        print(f"UNSTABLE at t={t:.0f}s"); break
    t += dt; nstep += 1

np.savez_compressed(os.environ.get("OUT", f"{ROOT}/sim/runs/swe2d_gorge_{RES:.0f}m.npz"),
                    peak_eta=peak_eta, peak_speed=peak_speed, Z=Z.astype(np.float32),
                    gt=np.array(gt), snaps=np.array(snaps), snap_t=np.array(snap_t))
print(f"done: {nstep} steps, {len(snaps)} snapshots; saved sim/runs/swe2d_gorge_{RES:.0f}m.npz", flush=True)

rows = list(csv.DictReader(open(f"{ROOT}/sim/inputs/trimline_profile.csv")))
diffs = []
for rr in rows:
    s = float(rr["chainage_m"])
    if not (CH0 <= s <= CH1): continue
    for side in ("L", "R"):
        tr_ = float(rr[f"trim_{side}_m"] or "nan")
        if not np.isfinite(tr_): continue
        i, j = rc(float(rr["x"]), float(rr["y"]))
        if not (0 <= i < NYg and 0 <= j < NXg): continue
        pe = peak_eta[max(0, i - 8):i + 9, max(0, j - 8):j + 9]
        pe = pe[np.isfinite(pe) & (pe > -1e30)]
        if len(pe) == 0: continue
        diffs.append(pe.max() - tr_)
d = np.array(diffs)
if len(d):
    print(f"model peak surface vs {len(d)} trimline elevations: "
          f"median diff {np.median(d):+.1f} m, IQR {np.percentile(d,25):+.1f}..{np.percentile(d,75):+.1f}")
