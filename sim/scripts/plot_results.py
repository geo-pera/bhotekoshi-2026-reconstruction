"""Results figure: peak-stage profile + trimline observations, and site stage series.

Run with the sim venv python (has matplotlib): sim/venv/bin/python plot_results.py [run.npz]
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
RUN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/sim/runs/route1d_V65_D30_nvar_debris.npz"
d = np.load(RUN)
ch = d["chainage"] / 1000.0
ps = d["peak_stage"]
t = d["t"] / 60.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))

ax1.plot(ch, ps, color="#1f77b4", lw=1.5, label="model peak stage (V=65 Mm$^3$, debris shape)")
ax1.axvline(20, color="gray", ls=":", lw=1)
ax1.text(20.3, ax1.get_ylim()[1] * 0.05 + 100, "injection", rotation=90, fontsize=8, color="gray")

# trimline observations
rows = list(csv.DictReader(open(f"{ROOT}/sim/inputs/trimline_profile.csv")))
tch, th, terr = [], [], []
for r in rows:
    for side in ("L", "R"):
        h = float(r[f"h_{side}_m"] or "nan")
        if np.isfinite(h) and h > 0.5:
            tch.append(float(r["chainage_m"]) / 1000.0)
            th.append(h)
            terr.append(20.0 if int(r[f"void_{side}"]) else 9.0)
if tch:
    ax1.errorbar(tch, th, yerr=terr, fmt="o", ms=4, color="#d62728", ecolor="#d62728",
                 elinewidth=0.8, capsize=2, label=f"trimline heights ({len(tch)} banks)", zorder=5)
for km, name in [(23.9, "Rasuwagadhi"), (38.9, "Syabrubesi"), (71.8, "Betrawati"), (108.0, "Galchhi")]:
    ax1.axvline(km, color="k", ls="--", lw=0.5, alpha=0.4)
    ax1.text(km + 0.5, ax1.get_ylim()[1] * 0.9, name, rotation=90, fontsize=7, alpha=0.7)
ax1.set_xlabel("chainage from avalanche source (km)")
ax1.set_ylabel("peak stage above thalweg (m)")
ax1.set_title("Bhote Koshi-Trishuli 2026-08-26: modeled peak stage vs trimline observations")
ax1.legend(fontsize=8)
ax1.set_xlim(0, 122); ax1.set_ylim(0, 150)
ax1.grid(alpha=0.25)

for key, style in [("Rasuwagadhi", "-"), ("Syabrubesi", "-"), ("Betrawati", "-"), ("Galchhi", "-")]:
    if key in d.files:
        ax2.plot(t, d[key] - d[key][0], style, lw=1.4, label=key)
ax2.axhline(9, color="gray", ls=":", lw=1)
ax2.text(300, 9.3, "Galchhi reported +9 m / 30 min", fontsize=8, color="gray")
ax2.set_xlabel("minutes after injection (~08:50 NPT)")
ax2.set_ylabel("stage rise (m)")
ax2.set_title("Stage rise at observation sites")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

fig.tight_layout()
out = f"{ROOT}/sim/runs/results_figure.png"
fig.savefig(out, dpi=140)
print("wrote", out)
