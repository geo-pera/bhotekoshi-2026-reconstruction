"""Forced-vortex velocity estimates at bends from trimline superelevation.

v = sqrt(g * Rc * dh / (k * W))   with k = 1 (Aberg et al. 2024: no correction
needed when the cross-flow surface inclination is measured directly, Fr 0.7-1.5).

Outer bank from signed centerline curvature; Rc from a 3-point circle fit over
+-SM m of centerline. Uncertainty: propagate trimline elevation errors
(+-9 m per bank, +-20 m where the DEM run touched a void).
"""
import csv
import numpy as np

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
SM = 300.0   # curvature half-window (m)
G = 9.81

cl = [(float(r["chainage_m"]), float(r["x_utm45"]), float(r["y_utm45"]))
      for r in csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv"))]
ch = np.array([c[0] for c in cl]); xs = np.array([c[1] for c in cl]); ys = np.array([c[2] for c in cl])

def circle_fit(s):
    """signed curvature via 3 points at s-SM, s, s+SM; sign>0 => turning left."""
    p = [(np.interp(t, ch, xs), np.interp(t, ch, ys)) for t in (s - SM, s, s + SM)]
    (x1, y1), (x2, y2), (x3, y3) = p
    a = np.hypot(x2 - x1, y2 - y1); b = np.hypot(x3 - x2, y3 - y2); c = np.hypot(x3 - x1, y3 - y1)
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    area2 = abs(cross)
    if area2 < 1e-6: return np.inf, 0
    R = a * b * c / (2 * area2)
    return R, np.sign(cross)

import os
rows = list(csv.DictReader(open(os.environ.get("TRIM_OUT", f"{ROOT}/sim/inputs/trimline_profile.csv"))))
print(f"{'km':>6} {'Rc(m)':>7} {'W(m)':>6} {'outer':>5} {'dh(m)':>6} {'v(m/s)':>7} {'range':>12} {'Fr':>5} flags")
results = []
for r in rows:
    tL, tR = float(r["trim_L_m"] or "nan"), float(r["trim_R_m"] or "nan")
    if not (np.isfinite(tL) and np.isfinite(tR)): continue
    dL, dR = float(r["d_L_m"]), float(r["d_R_m"])
    s = float(r["chainage_m"])
    Rc, sgn = circle_fit(s)
    if not np.isfinite(Rc) or Rc > 5000 or sgn == 0: continue   # effectively straight
    W = dL + dR
    if W < 40: continue
    # normal (nxv,nyv) points LEFT of flow; turning left (sgn>0) => outer bank RIGHT
    outer, inner = ("R", "L") if sgn > 0 else ("L", "R")
    dh = (tR - tL) if outer == "R" else (tL - tR)
    if dh <= 0:
        results.append((s, Rc, W, outer, dh, np.nan, (np.nan, np.nan), np.nan, "dh<=0 (no superelev signal)"))
        continue
    v = np.sqrt(G * Rc * dh / W)
    # uncertainty: sigma_dh from both banks
    sig = np.hypot(20.0 if int(r["void_L"]) else 9.0, 20.0 if int(r["void_R"]) else 9.0)
    dh_lo, dh_hi = max(dh - sig, 0.5), dh + sig
    v_lo = np.sqrt(G * (Rc * 0.7) * dh_lo / (W * 1.3))
    v_hi = np.sqrt(G * (Rc * 1.3) * dh_hi / (W * 0.7))
    hmean = np.nanmean([float(r["h_L_m"]), float(r["h_R_m"])])
    Fr = v / np.sqrt(G * hmean) if hmean > 0 else np.nan
    flags = []
    if int(r["void_L"]) or int(r["void_R"]): flags.append("void-DEM")
    if Fr < 0.5 or Fr > 2.5: flags.append("Fr-outside-validated-range")
    results.append((s, Rc, W, outer, dh, v, (v_lo, v_hi), Fr, ",".join(flags) or "-"))

for s, Rc, W, outer, dh, v, (vlo, vhi), Fr, flags in results:
    vtxt = f"{v:7.1f}" if np.isfinite(v) else "      -"
    rtxt = f"{vlo:4.1f}-{vhi:5.1f}" if np.isfinite(v) else "           -"
    print(f"{s/1000:6.1f} {Rc:7.0f} {W:6.0f} {outer:>5} {dh:6.1f} {vtxt} {rtxt} {Fr:5.2f} {flags}")

with open(f"{ROOT}/sim/inputs/superelevation_velocities.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["chainage_m", "Rc_m", "W_m", "outer_bank", "dh_m", "v_ms", "v_lo", "v_hi", "Froude", "flags"])
    for s, Rc, W, outer, dh, v, (vlo, vhi), Fr, flags in results:
        w.writerow([f"{s:.0f}", f"{Rc:.0f}", f"{W:.0f}", outer, f"{dh:.1f}",
                    f"{v:.1f}" if np.isfinite(v) else "", f"{vlo:.1f}" if np.isfinite(v) else "",
                    f"{vhi:.1f}" if np.isfinite(v) else "", f"{Fr:.2f}" if np.isfinite(Fr) else "", flags])
print("wrote sim/inputs/superelevation_velocities.csv")
