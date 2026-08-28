"""Animate the routed flood wave along the corridor -> GIF."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
d = np.load(f"{ROOT}/sim/runs/route1d_V100_D30_nvar1.0_debris.npz")
ch = d["chainage"] / 1000.0; bed = d["bed"]
st = d["snap_t"]; ss = d["snap_stage"]
# subsample frames to ~180
step = max(1, len(st) // 180)
st = st[::step]; ss = ss[::step]

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=90)
ax.fill_between(ch, 0, bed, color="#8a7f70", zorder=1)
ax.plot(ch, bed, color="#5d5347", lw=0.8, zorder=2)
water = ax.fill_between(ch, bed, bed, color="#2f7ed8", zorder=3)
for km, name in [(23.9, "Rasuwagadhi"), (38.9, "Syabrubesi"), (71.8, "Betrawati"), (108.0, "Galchhi")]:
    ax.axvline(km, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.text(km, 5350, name, rotation=90, fontsize=8, ha="right", va="top", alpha=0.75)
tlabel = ax.text(0.985, 0.94, "", transform=ax.transAxes, ha="right", fontsize=13,
                 fontweight="bold", color="#1a4a7a")
ax.set_xlim(0, 122); ax.set_ylim(200, 5500)
ax.set_xlabel("chainage from avalanche source (km)")
ax.set_ylabel("elevation (m)")
ax.set_title("Bhote Koshi–Trishuli outburst flood, 26 Aug 2026 — routed wave (V=100 Mm³)\n"
             "water depth exaggerated 20× for visibility", fontsize=10)

def update(i):
    global water
    water.remove()
    surf = bed + 20 * ss[i]          # 20x exaggeration
    water = ax.fill_between(ch, bed, np.maximum(surf, bed), color="#2f7ed8", zorder=3)
    hh = int((st[i]) // 3600); mm = int((st[i]) % 3600 // 60)
    clock_min = 8 * 60 + 50 + st[i] / 60   # injection ~08:50 NPT
    ax.set_ylim(200, 5500)
    tlabel.set_text(f"~{int(clock_min//60):02d}:{int(clock_min%60):02d} NPT  (t+{hh:d}h{mm:02d}m)")
    return water, tlabel

anim = FuncAnimation(fig, update, frames=len(st), blit=False)
out = f"{ROOT}/sim/runs/flood_wave_1d.gif"
anim.save(out, writer=PillowWriter(fps=15))
print("wrote", out)
