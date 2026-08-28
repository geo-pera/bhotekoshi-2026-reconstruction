"""Animate the 2D gorge flood with impact layer: OSM buildings/roads flip red as
the wave hits them; sediment scar persists where water has passed; running toll."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from rasterio.warp import transform as rio_transform

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
d = np.load(f"{ROOT}/sim/runs/swe2d_gorge_24m.npz")
Z = d["Z"]; snaps = d["snaps"]; st = d["snap_t"]; gt = d["gt"]
NYg, NXg = Z.shape
RES = gt[1]

# hillshade
gy, gx = np.gradient(Z.astype(float), RES)
az, alt = np.radians(315), np.radians(45)
slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
hs = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)

# OSM
osm = json.load(open(f"{ROOT}/sim/inputs/osm_gorge.json"))["elements"]
nodes = {e["id"]: (e["lon"], e["lat"]) for e in osm if e["type"] == "node"}
def to_grid(lon, lat):
    xs_, ys_ = rio_transform("EPSG:4326", "EPSG:32645", [lon], [lat])
    return (ys_[0] - gt[3]) / gt[5], (xs_[0] - gt[0]) / gt[1]   # row, col (float)

blds = []
road_lines = []; road_pts = []
for e in osm:
    if e["type"] != "way": continue
    tags = e.get("tags", {})
    pts = [to_grid(*nodes[n]) for n in e["nodes"] if n in nodes]
    pts = [(r_, c_) for r_, c_ in pts if 0 <= r_ < NYg and 0 <= c_ < NXg]
    if not pts: continue
    if "building" in tags:
        rr = sum(p[0] for p in pts) / len(pts); cc = sum(p[1] for p in pts) / len(pts)
        blds.append((rr, cc))
    elif "highway" in tags:
        road_lines.append(pts)
        for (r1, c1), (r2, c2) in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.hypot(r2 - r1, c2 - c1)))
            for k in range(n):
                road_pts.append((r1 + (r2 - r1) * k / n, c1 + (c2 - c1) * k / n))
blds = np.array(blds); road_pts = np.array(road_pts)
print(f"{len(blds)} buildings, {len(road_pts)} road samples")

# per-frame cumulative impact
WETD = 0.5
bi = blds.astype(int); ri = road_pts.astype(int)
wetmax = np.zeros_like(Z, dtype=np.float32)
bld_hit_frame = np.full(len(blds), 10 ** 9)
road_wet_frac = []
scars = []
for i, s in enumerate(snaps):
    wetmax = np.maximum(wetmax, s)
    hit = wetmax[bi[:, 0], bi[:, 1]] > WETD
    bld_hit_frame[hit & (bld_hit_frame > i)] = i
    road_wet_frac.append((wetmax[ri[:, 0], ri[:, 1]] > WETD).mean())
    scars.append(((wetmax > WETD) & (s < 0.3)))
road_km_total = len(road_pts) * RES / 1000.0

stops = [(0.00, (0.66, 0.90, 0.94, 0.55)), (0.25, (0.36, 0.75, 0.86, 0.78)),
         (0.50, (0.18, 0.49, 0.78, 0.90)), (0.75, (0.10, 0.29, 0.61, 0.95)),
         (1.00, (0.04, 0.13, 0.38, 0.97))]
wcmap = LinearSegmentedColormap.from_list("w", stops); wcmap.set_over((0.03, 0.08, 0.24, 1.0))
VMAX = float(np.percentile(snaps[snaps > 0.3], 98))
scar_cmap = ListedColormap([(0.72, 0.58, 0.42, 0.75)])

fig, ax = plt.subplots(figsize=(6.4, 8.4), dpi=100)
fig.subplots_adjust(left=0.02, right=0.86, top=0.90, bottom=0.02)
ax.imshow(hs, cmap="gray", vmin=-0.15, vmax=1.25)
scar_im = ax.imshow(np.ma.masked_where(~scars[0], np.ones_like(Z)), cmap=scar_cmap, vmin=0, vmax=1)
im = ax.imshow(np.ma.masked_less(snaps[0], 0.3), cmap=wcmap, vmin=0, vmax=VMAX)
fig.colorbar(im, ax=ax, shrink=0.55, pad=0.03, label="water depth (m)", extend="max")
for pts in road_lines:
    ax.plot([p[1] for p in pts], [p[0] for p in pts], color="#4a4238", lw=0.5, alpha=0.8, zorder=4)
intact = ax.scatter(blds[:, 1], blds[:, 0], s=2.5, c="#f5f0e6", edgecolors="#55503f",
                    linewidths=0.2, zorder=5)
gone = ax.scatter([], [], s=7, c="#e8321e", edgecolors="#7a0f04", linewidths=0.3, zorder=6)
# Syabrubesi label
r_s, c_s = to_grid(85.348, 28.160)
ax.annotate("Syabrubesi", (c_s, r_s), textcoords="offset points", xytext=(14, 8),
            fontsize=9, fontweight="bold", color="#231f14",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2))
tl = ax.text(0.02, 0.985, "", transform=ax.transAxes, fontsize=12, fontweight="bold",
             color="#08306b", va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
toll = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=11, fontweight="bold",
               color="#a51205", va="bottom", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
fig.suptitle("Syabrubesi gorge — simulated flood, 26 Aug 2026", fontsize=11, y=0.965)
ax.set_xticks([]); ax.set_yticks([])
km2 = 2000 / RES
ax.plot([NXg - 15 - km2, NXg - 15], [NYg - 12, NYg - 12], color="k", lw=2)
ax.text(NXg - 15 - km2 / 2, NYg - 18, "2 km", ha="center", fontsize=8)

def update(i):
    im.set_data(np.ma.masked_less(snaps[i], 0.3))
    scar_im.set_data(np.ma.masked_where(~scars[i], np.ones_like(Z)))
    hitmask = bld_hit_frame <= i
    gone.set_offsets(blds[hitmask][:, ::-1] if hitmask.any() else np.empty((0, 2)))
    intact.set_offsets(blds[~hitmask][:, ::-1])
    clock = 8 * 60 + 50 + st[i] / 60
    tl.set_text(f"~{int(clock//60):02d}:{int(clock%60):02d} NPT")
    toll.set_text(f"structures in flooded area: {hitmask.sum():,}\n"
                  f"road inundated: {road_wet_frac[i]*road_km_total:.1f} km")
    return im, scar_im, gone, intact, tl, toll

anim = FuncAnimation(fig, update, frames=len(snaps), blit=False)
out = f"{ROOT}/sim/runs/flood_gorge_2d.gif"
anim.save(out, writer=PillowWriter(fps=12))
print("wrote", out, f"| final toll: {(bld_hit_frame<10**9).sum()} structures, "
      f"{road_wet_frac[-1]*road_km_total:.1f} km road")
