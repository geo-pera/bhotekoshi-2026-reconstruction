"""3D oblique animation of the flood wall through a 2 km gorge slice."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, LightSource

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
d = np.load(f"{ROOT}/sim/runs/swe2d_gorge_24m.npz")
Z = d["Z"].astype(float); snaps = d["snaps"]; st = d["snap_t"]; gt = d["gt"]
RES = gt[1]

# slice: km 35.3-37.3 along the centerline
cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
ch = np.array([float(c["chainage_m"]) for c in cl])
xs = np.array([float(c["x_utm45"]) for c in cl]); ys = np.array([float(c["y_utm45"]) for c in cl])
cx = np.interp(36300, ch, xs); cy = np.interp(36300, ch, ys)
ci = int((cy - gt[3]) / gt[5]); cj = int((cx - gt[0]) / gt[1])
HL_ROW = int(750 / RES); HL_COL = int(480 / RES)   # 1.5 km along-valley, ~1 km across
r0, r1 = ci - HL_ROW, ci + HL_ROW
c0, c1 = cj - HL_COL, cj + HL_COL
Zs = Z[r0:r1, c0:c1]
ny, nx = Zs.shape
def up2(a):
    """2x bilinear upsample"""
    ii = (np.arange(a.shape[0] * 2) / 2.0).clip(0, a.shape[0] - 1.001)
    jj = (np.arange(a.shape[1] * 2) / 2.0).clip(0, a.shape[1] - 1.001)
    i0 = ii.astype(int); j0 = jj.astype(int)
    fi = (ii - i0)[:, None]; fj = (jj - j0)[None, :]
    A = a[np.ix_(i0, j0)]; B = a[np.ix_(i0 + 1, j0)]
    C = a[np.ix_(i0, j0 + 1)]; D = a[np.ix_(i0 + 1, j0 + 1)]
    return A * (1 - fi) * (1 - fj) + B * fi * (1 - fj) + C * (1 - fi) * fj + D * fi * fj
Zs = up2(Zs); ny, nx = Zs.shape
RESR = RES / 2.0
X, Y = np.meshgrid(np.arange(nx) * RESR, np.arange(ny) * RESR)
zmin = Zs.min()

wramp = LinearSegmentedColormap.from_list("w", ["#7fd0e8", "#2f7ec8", "#0b2260"])
ls = LightSource(azdeg=315, altdeg=50)
terrain_rgb = ls.shade(Zs, cmap=plt.get_cmap("gist_earth"), vert_exag=1.2,
                       blend_mode="soft", vmin=zmin - 200, vmax=Zs.max() + 400)

frames = [i for i in range(len(snaps)) if 13 * 60 <= st[i] <= 42 * 60]
print(f"{len(frames)} frames, slice {nx}x{ny} at {RES} m, relief {zmin:.0f}-{Zs.max():.0f} m")

fig = plt.figure(figsize=(8.8, 5.8), dpi=100)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
ax.set_position([-0.22, -0.24, 1.44, 1.46])

def draw(i):
    ax.clear()
    dep = up2(snaps[i][r0:r1, c0:c1].astype(float))
    wet = dep > 0.4
    surf = np.where(wet, Zs + dep, Zs)
    fc = terrain_rgb.copy()
    dnorm = np.clip(dep / 45.0, 0, 1)
    wcol = wramp(dnorm)
    fc[wet] = wcol[wet]
    ax.plot_surface(X, Y, surf, facecolors=fc, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    # 40 m reference post on the bank near mid-slice
    pr, pc = ny // 2, int(nx * 0.62)
    zb = Zs[pr, pc]
    ax.plot([X[pr, pc]] * 2, [Y[pr, pc]] * 2, [zb, zb + 40], color="#ff2d00", lw=3)
    ax.text(X[pr, pc] + 30, Y[pr, pc], zb + 46, "40 m\n(~13 storeys)", color="#a51205",
            fontsize=8, fontweight="bold")
    clock = 8 * 60 + 50 + st[i] / 60
    clock_txt.set_text(f"~{int(clock//60):02d}:{int(clock%60):02d} NPT")
    ax.set_zlim(zmin - 10, zmin + 430)
    ax.set_xlim(0, nx * RESR); ax.set_ylim(0, ny * RESR)
    ax.set_box_aspect((nx * RESR, ny * RESR, 620))
    ax.view_init(elev=14, azim=88)
    ax.set_axis_off()

clock_txt = fig.text(0.04, 0.92, "", fontsize=14, fontweight="bold", color="#08306b")
fig.text(0.04, 0.03, "Bhote Koshi gorge, km 35.3–36.8 — looking upstream; wave arrives from the far end",
         fontsize=8.5, color="#333333")
anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames), blit=False)
out = f"{ROOT}/sim/runs/flood_wall_3d.gif"
anim.save(out, writer=PillowWriter(fps=7))
print("wrote", out)
