"""Build the single full-release archive: dist/bhotekoshi-2026-full-release.zip"""
import os, zipfile, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = f"{ROOT}/dist"; os.makedirs(DIST, exist_ok=True)
zp = f"{DIST}/bhotekoshi-2026-full-release.zip"
if os.path.exists(zp): os.remove(zp)
n = 0
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
    for d in ("dist/bhotekoshi-2026-flood-data", "nepal-flash-flood-2026-08-26", "vantor",
              "terrain", "s2", "sim/dem", "sim/masks"):
        for root, _, files in os.walk(f"{ROOT}/{d}"):
            if "venv" in root or root.endswith("/work"): continue
            for fn in files:
                if fn == ".DS_Store": continue
                p = os.path.join(root, fn)
                arc = os.path.relpath(p, ROOT).replace("dist/bhotekoshi-2026-flood-data", "derived")
                z.write(p, f"bhotekoshi-2026-full-release/{arc}"); n += 1
print(f"zip: {os.path.getsize(zp)/1e9:.1f} GB, {n} files")
h = hashlib.sha256()
with open(zp, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
        h.update(chunk)
with open(f"{DIST}/MANIFEST.md", "w") as m:
    m.write("# Bhote Koshi 2026 — full data release\n\n"
            f"bhotekoshi-2026-full-release.zip — {os.path.getsize(zp)/1e9:.2f} GB\n"
            f"sha256: {h.hexdigest()}\n\n"
            "Layout: derived/ (measurements, models, scenario, docs, viewer) · "
            "nepal-flash-flood-2026-08-26/ (Planet imagery, STAC) · vantor/ (WorldView) · "
            "terrain/ s2/ sim/dem sim/masks (rasters)\n\n"
            "Imagery (c) Planet Labs PBC / (c) Vantor, open data programs, CC BY-NC 4.0. "
            "Terrain: NASA NSIDC HMA DEM, Copernicus GLO-30, Copernicus Sentinel-2.\n")
print("manifest written")
