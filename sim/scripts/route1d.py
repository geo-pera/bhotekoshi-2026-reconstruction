"""1D Saint-Venant routing of the surge down the Bhote Koshi-Trishuli corridor.

Finite-volume (A,Q) on the cross-section tables from xsections.npz, Rusanov flux
for advection, surface-gradient pressure term, semi-implicit Manning friction.
Inject a parametric hydrograph just above the Lende-Bhote Koshi confluence and
record stage series at the observation points.

Usage: route1d.py [volume_Mm3] [duration_min] [manning_n]
"""
import sys, os
import numpy as np

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
XSEC = sys.argv[5] if len(sys.argv) > 5 else f"{ROOT}/sim/inputs/xsections.npz"
d = np.load(XSEC)
chain = d["chainage"].astype(np.float64)
bed = d["thalweg"].astype(np.float64)
stage_ax = d["stage"].astype(np.float64)          # 0..120 m
A_tab = d["A"].astype(np.float64)                 # (nsec, nstage)
B_tab = d["B"].astype(np.float64)

# fill empty sections (void thalweg) from neighbors
empty = A_tab[:, -1] <= 0
if empty.any():
    idx = np.arange(len(chain)); oki = idx[~empty]
    for j in range(A_tab.shape[1]):
        A_tab[empty, j] = np.interp(idx[empty], oki, A_tab[~empty, j])
        B_tab[empty, j] = np.interp(idx[empty], oki, B_tab[~empty, j])
print(f"filled {empty.sum()} empty sections from neighbors")

N = len(chain)
DX = float(chain[1] - chain[0])
G = 9.81

VOL = float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else 20e6     # m^3
DUR = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 1800.0    # s
_narg = sys.argv[3] if len(sys.argv) > 3 else "var"
NMAN = _narg
SHAPE = sys.argv[4] if len(sys.argv) > 4 else "triangle"
QP = 2 * VOL / DUR                                                 # triangular peak
T_RISE = DUR / 6
INJ = int(np.searchsorted(chain, float(os.environ.get('INJECT_M', 20000))))
T_END = 6.0 * 3600
if NMAN == "var":
    nprof = np.where(chain < 39000, 0.10, np.where(chain < 72000, 0.05, 0.04))
else:
    nprof = np.full(len(chain), float(NMAN))
NSC = float(os.environ.get("N_SCALE", "1.0"))
nprof = nprof * NSC
print(f"scenario: V={VOL/1e6:.0f} Mm3, duration {DUR/60:.0f} min, Qp={QP:.0f} m3/s, n={NMAN} x{NSC}")

# min area floor so tables invert cleanly
A_tab = np.maximum(A_tab, 1e-3)
def stage_of_A(Avec):
    s = np.empty(N)
    for i in range(N):
        s[i] = np.interp(Avec[i], A_tab[i], stage_ax)
    return s
def B_of_stage(svec):
    j = np.clip(svec, 0, stage_ax[-1])
    return np.array([np.interp(j[i], stage_ax, B_tab[i]) for i in range(N)])

# initial condition: token wet channel
h0 = 0.5
A = np.array([np.interp(h0, stage_ax, A_tab[i]) for i in range(N)])
Q = np.zeros(N)

def _tri(t):
    if t < 0 or t >= DUR: return 0.0
    return QP * (t / T_RISE if t < T_RISE else 1 - (t - T_RISE) / (DUR - T_RISE))
def _gamma_shape(t, tp, k):
    if not np.isfinite(t) or t <= 0: return 0.0
    x = t / tp
    b = max(x * float(np.exp(np.clip(1 - x, -700, 50))), 0.0)
    return b ** k
if SHAPE == "triangle":
    hydrograph = _tri
elif SHAPE == "debris":
    # steep bore: rise 3 min, heavy tail; normalize to VOL
    tp, k = 180.0, 1.2
    tt = np.arange(0, 4 * DUR, 5.0)
    raw = np.array([_gamma_shape(x, tp, k) for x in tt])
    scale = VOL / np.trapezoid(raw, tt)
    hydrograph = lambda t, _s=scale, _tp=tp, _k=k: _s * _gamma_shape(t, _tp, _k)
elif SHAPE == "breach":
    # erosional breach: slow start, peak at ~20 min, quick fall
    tp, k = 1200.0, 3.0
    tt = np.arange(0, 4 * DUR, 5.0)
    raw = np.array([_gamma_shape(x, tp, k) for x in tt])
    scale = VOL / np.trapezoid(raw, tt)
    hydrograph = lambda t, _s=scale, _tp=tp, _k=k: _s * _gamma_shape(t, _tp, _k)
elif SHAPE == "instant":
    # sudden dam collapse: rise 1 min, exp decay tau=8 min
    tau = 480.0
    Qp_i = VOL / (60.0 / 2 + tau)   # approx normalization (triangle rise + exp tail)
    hydrograph = lambda t: 0.0 if t < 0 else (Qp_i * t / 60 if t < 60 else Qp_i * np.exp(-(t - 60) / tau))
else:
    raise SystemExit(f"unknown shape {SHAPE}")
_pk = max(hydrograph(x) for x in np.arange(0, 3 * DUR, 10.0))
print(f"shape={SHAPE}: peak inflow {_pk:.0f} m3/s")

# observation points
from osgeo import osr
sr = osr.SpatialReference(); sr.ImportFromEPSG(4326); sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
srU = osr.SpatialReference(); srU.ImportFromEPSG(32645); srU.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
tr = osr.CoordinateTransformation(sr, srU)
X = d["x"]; Y = d["y"]
def station_near(lon, lat):
    x, y, _ = tr.TransformPoint(lon, lat)
    return int(np.argmin((X - x) ** 2 + (Y - y) ** 2))
OBS = {"Rasuwagadhi(23.9km)": int(np.searchsorted(chain, 23900)),
       "Syabrubesi(38.9km)": int(np.searchsorted(chain, 38900)),
       "Betrawati(71.8km)": int(np.searchsorted(chain, 71800)),
       "Galchhi": station_near(84.995, 27.800)}
print("Galchhi station at chainage", f"{chain[OBS['Galchhi']]/1000:.1f} km")

rec_t = []; rec = {k: [] for k in OBS}
SNAPS = os.environ.get("SAVE_SNAPS")
snap_t = []; snap_stage = []; next_snap = 0.0
I33 = int(np.searchsorted(chain, 33000)); rec_q33 = []
peak_stage = np.zeros(N)
t = 0.0; nstep = 0
while t < T_END:
    stg = stage_of_A(A)
    B = np.maximum(B_of_stage(stg), 5.0)
    h = np.maximum(A / B, 0.02)
    V = Q / A
    V = np.clip(V, -60, 60)
    c = np.sqrt(G * h)
    lam = np.abs(V) + c
    dt = min(0.3 * DX / max(lam.max(), 1.0), 2.0)
    if not np.isfinite(dt) or dt <= 0:
        print(f"BLEW UP (dt) at t={t:.0f}s step {nstep}"); break
    eta = bed + stg
    # Rusanov fluxes at interfaces i+1/2 for mass and advective momentum
    aL = lam[:-1]; aR = lam[1:]; amax = np.maximum(aL, aR)
    Fmass = 0.5 * (Q[:-1] + Q[1:]) - 0.5 * amax * (A[1:] - A[:-1])
    adv = Q * V
    Fmom = 0.5 * (adv[:-1] + adv[1:]) - 0.5 * amax * (Q[1:] - Q[:-1])
    # updates
    An = A.copy(); Qn = Q.copy()
    An[1:-1] -= dt / DX * (Fmass[1:] - Fmass[:-1])
    Qn[1:-1] -= dt / DX * (Fmom[1:] - Fmom[:-1])
    # pressure/gravity: -g A d(eta)/dx (centered)
    detadx = np.zeros(N)
    detadx[1:-1] = (eta[2:] - eta[:-2]) / (2 * DX)
    Qn[1:-1] -= dt * G * A[1:-1] * detadx[1:-1]
    # friction semi-implicit, R ~ A/B
    R = np.maximum(A / B, 0.05)
    fr = G * nprof ** 2 * np.abs(V) / R ** (4.0 / 3.0)
    Qn[1:-1] /= (1 + dt * fr[1:-1])
    # inflow: add hydrograph as source at INJ
    qin = hydrograph(t)
    An[INJ-1:INJ+2] += dt * qin / (3 * DX)
    # boundaries: upstream closed, downstream transmissive
    An[0] = An[1]; Qn[0] = 0.0
    An[-1] = An[-2]
    S_out = max((bed[-2] - bed[-1]) / DX, 1e-3)
    R_out = max(An[-1] / max(B[-1], 5.0), 0.05)
    Qn[-1] = An[-1] * (1.0 / nprof[-1]) * R_out ** (2.0 / 3.0) * np.sqrt(S_out)
    A = np.maximum(An, 1e-3); Q = Qn
    peak_stage = np.maximum(peak_stage, stage_of_A(A)) if nstep % 20 == 0 else peak_stage
    if SNAPS and t >= next_snap:
        snap_t.append(t); snap_stage.append(stage_of_A(A).astype(np.float32)); next_snap += 60.0
    if nstep % 60 == 0:
        rec_t.append(t)
        stg_now = stage_of_A(A)
        for k, i in OBS.items(): rec[k].append(stg_now[i])
        rec_q33.append(Q[I33])
    t += dt; nstep += 1
    if not np.isfinite(A).all() or not np.isfinite(Q).all():
        print(f"BLEW UP at t={t:.0f}s step {nstep}"); break

rec_t = np.array(rec_t)
print(f"\nran {nstep} steps to t={t/60:.0f} min")
print(f"{'site':<22}{'arrive(+0.5m)':>14}{'peak stage':>12}{'t_peak':>8}{'rise 0.5->90%':>14}{'max 30-min rise':>17}")
for k in OBS:
    s = np.array(rec[k]); s0 = s[0]
    rise = s - s0
    ia = np.argmax(rise > 0.5) if (rise > 0.5).any() else -1
    tpk = rec_t[np.argmax(s)] / 60
    m30 = 0.0
    for j in range(len(s)):
        j2 = np.searchsorted(rec_t, rec_t[j] + 1800)
        if j2 < len(s): m30 = max(m30, s[j2] - s[j])
    at = f"{rec_t[ia]/60:6.0f} min" if ia >= 0 else "     -"
    r90 = "-"
    if ia >= 0:
        i90 = np.argmax(rise >= 0.9 * np.max(rise))
        if i90 > ia: r90 = f"{(rec_t[i90]-rec_t[ia])/60:.0f} min"
    print(f"{k:<22}{at:>14}{np.max(rise):>10.1f} m{tpk:>7.0f}m{r90:>14}{m30:>15.1f} m")
np.savez_compressed(f"{ROOT}/sim/runs/route1d_V{VOL/1e6:.0f}_D{DUR/60:.0f}_n{_narg}{NSC}_{SHAPE}.npz",
                    t=rec_t, **{k.split('(')[0]: np.array(v) for k, v in rec.items()},
                    peak_stage=peak_stage, chainage=chain, Q33=np.array(rec_q33),
                    snap_t=np.array(snap_t), snap_stage=np.array(snap_stage) if snap_stage else np.zeros(0),
                    bed=bed)
print("saved run to sim/runs/")
