"""3D slice at Syabrubesi: flood wall arrives, buildings flip red as inundated."""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, LightSource
from rasterio.warp import transform as rio_transform

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
d = np.load(f"{ROOT}/sim/runs/swe2d_gorge_24m.npz")
Z = d["Z"].astype(float); snaps = d["snaps"]; st = d["snap_t"]; gt = d["gt"]
RES = gt[1]

cl = list(csv.DictReader(open(f"{ROOT}/sim/inputs/centerline_v3.csv")))
ch = np.array([float(c["chainage_m"]) for c in cl])
xs = np.array([float(c["x_utm45"]) for c in cl]); ys = np.array([float(c["y_utm45"]) for c in cl])
CHC = 39150.0
cx = np.interp(CHC, ch, xs); cy = np.interp(CHC, ch, ys)
ci = int((cy - gt[3]) / gt[5]); cj = int((cx - gt[0]) / gt[1])
HL_ROW = int(850 / RES); HL_COL = int(750 / RES)
r0, r1 = ci - HL_ROW, ci + HL_ROW
c0, c1 = cj - HL_COL, cj + HL_COL
Zs0 = Z[r0:r1, c0:c1]

def up2(a):
    ii = (np.arange(a.shape[0] * 2) / 2.0).clip(0, a.shape[0] - 1.001)
    jj = (np.arange(a.shape[1] * 2) / 2.0).clip(0, a.shape[1] - 1.001)
    i0 = ii.astype(int); j0 = jj.astype(int)
    fi = (ii - i0)[:, None]; fj = (jj - j0)[None, :]
    A = a[np.ix_(i0, j0)]; B = a[np.ix_(i0 + 1, j0)]
    C = a[np.ix_(i0, j0 + 1)]; D = a[np.ix_(i0 + 1, j0 + 1)]
    return A * (1 - fi) * (1 - fj) + B * fi * (1 - fj) + C * (1 - fi) * fj + D * fi * fj

Zs = up2(Zs0); ny, nx = Zs.shape
RESR = RES / 2.0
X, Y = np.meshgrid(np.arange(nx) * RESR, np.arange(ny) * RESR)
zmin = Zs.min()

# buildings inside slice
osm = json.load(open(f"{ROOT}/sim/inputs/osm_gorge.json"))["elements"]
nodes = {e["id"]: (e["lon"], e["lat"]) for e in osm if e["type"] == "node"}
blds = []
for e in osm:
    if e["type"] != "way" or "building" not in e.get("tags", {}): continue
    pts = [nodes[n] for n in e["nodes"] if n in nodes]
    if not pts: continue
    lon = sum(p[0] for p in pts) / len(pts); lat = sum(p[1] for p in pts) / len(pts)
    xu, yu = rio_transform("EPSG:4326", "EPSG:32645", [lon], [lat])
    gr = (yu[0] - gt[3]) / gt[5]; gc = (xu[0] - gt[0]) / gt[1]
    if r0 + 1 <= gr < r1 - 1 and c0 + 1 <= gc < c1 - 1:
        lr = (gr - r0) * 2; lc = (gc - c0) * 2      # upsampled local indices
        blds.append((lr, lc, int(gr), int(gc)))
print(f"{len(blds)} buildings in slice")

wramp = LinearSegmentedColormap.from_list("w", ["#7fd0e8", "#2f7ec8", "#0b2260"])
ls = LightSource(azdeg=315, altdeg=50)
terrain_rgb = ls.shade(Zs, cmap=plt.get_cmap("gist_earth"), vert_exag=1.2,
                       blend_mode="soft", vmin=zmin - 200, vmax=Zs.max() + 400)

frames = [i for i in range(len(snaps)) if 15 * 60 <= st[i] <= 46 * 60]
wetmax_hist = []
wm = np.zeros_like(Z, dtype=np.float32)
for i in range(len(snaps)):
    wm = np.maximum(wm, snaps[i])
    wetmax_hist.append(wm.copy())

fig = plt.figure(figsize=(8.8, 5.8), dpi=100)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
ax.set_position([-0.22, -0.24, 1.44, 1.46])
clock_txt = fig.text(0.04, 0.92, "", fontsize=14, fontweight="bold", color="#08306b")
toll_txt = fig.text(0.04, 0.86, "", fontsize=11, fontweight="bold", color="#a51205")
fig.text(0.04, 0.03, "Syabrubesi confluence (km 38.3–40.0) — buildings turn red once inundated (OSM footprints, simulated flood)",
         fontsize=8.5, color="#333333")

BH = 6.5   # 2-storey village house to eaves (m)
BSZ = 10.0
def bilin(a, r, c):
    r = min(max(r, 0), a.shape[0] - 1.001); c = min(max(c, 0), a.shape[1] - 1.001)
    i0, j0 = int(r), int(c); fr, fc = r - i0, c - j0
    return (a[i0, j0] * (1-fr) * (1-fc) + a[i0+1, j0] * fr * (1-fc)
            + a[i0, j0+1] * (1-fr) * fc + a[i0+1, j0+1] * fr * fc)
def draw(i):
    ax.clear()
    dep = up2(snaps[i][r0:r1, c0:c1].astype(float))
    wet = dep > 0.4
    surf = np.where(wet, Zs + dep, Zs)
    fc = terrain_rgb.copy()
    wcol = wramp(np.clip(dep / 45.0, 0, 1))
    fc[wet] = wcol[wet]
    ax.plot_surface(X, Y, surf, facecolors=fc, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    hit = 0
    groups = {"intact": [], "wet": [], "wreck": []}
    for lr, lc, gr, gc in blds:
        zb = bilin(Zs, lr, lc) - 1.0          # embed base 1 m into terrain
        dnow = bilin(dep, lr, lc)
        ever = wetmax_hist[i][gr, gc] > 0.5
        if ever: hit += 1
        if dnow > 0.3 and zb + dnow >= zb + BH:      # water over the eaves: covered, not drawn
            continue
        elif dnow > 0.3:                              # in the water, partially exposed
            groups["wet"].append((lc * RESR - BSZ/2, lr * RESR - BSZ/2, zb))
        elif ever:                                    # water has receded: wreckage
            groups["wreck"].append((lc * RESR - BSZ/2, lr * RESR - BSZ/2, zb))
        else:
            groups["intact"].append((lc * RESR - BSZ/2, lr * RESR - BSZ/2, zb))
    for key, col, ecol in (("intact", "#f5f0e6", "#6b6350"),
                           ("wet", "#7a1508", "#3f0a03"),
                           ("wreck", "#e8321e", "#7a0f04")):
        g = groups[key]
        if g:
            bx, by, bz = zip(*g)
            ax.bar3d(list(bx), list(by), list(bz), BSZ, BSZ, BH, color=col,
                     edgecolor=ecol, linewidth=0.2, shade=True)
    pr, pc = int(ny * 0.55), int(nx * 0.30)
    zb = Zs[pr, pc]
    ax.plot([X[pr, pc]] * 2, [Y[pr, pc]] * 2, [zb, zb + 40], color="#ff2d00", lw=3)
    ax.text(X[pr, pc] + 25, Y[pr, pc], zb + 46, "40 m", color="#a51205", fontsize=8, fontweight="bold")
    clock = 8 * 60 + 50 + st[i] / 60
    clock_txt.set_text(f"~{int(clock//60):02d}:{int(clock%60):02d} NPT")
    toll_txt.set_text(f"buildings hit so far: {hit}/{len(blds)}")
    ax.set_zlim(zmin - 10, zmin + 430)
    ax.set_xlim(0, nx * RESR); ax.set_ylim(0, ny * RESR)
    ax.set_box_aspect((nx * RESR, ny * RESR, 560))
    ax.view_init(elev=22, azim=115)
    ax.set_axis_off()

anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames), blit=False)
out = f"{ROOT}/sim/runs/flood_syabrubesi_3d.gif"
anim.save(out, writer=PillowWriter(fps=7))
print("wrote", out)
