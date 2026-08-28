"""Full package: imagery + raster archives alongside the derived-data zip."""
import os, zipfile, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = f"{ROOT}/dist"
os.makedirs(DIST, exist_ok=True)

SETS = {
    "bhotekoshi-2026-imagery-planet.zip": ["nepal-flash-flood-2026-08-26"],
    "bhotekoshi-2026-imagery-vantor.zip": ["vantor"],
    "bhotekoshi-2026-terrain-masks.zip": ["terrain", "s2", "sim/dem", "sim/masks"],
}
def build(zname, dirs):
    zp = f"{DIST}/{zname}"
    if os.path.exists(zp): os.remove(zp)
    n = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for d in dirs:
            base = f"{ROOT}/{d}"
            for root, subdirs, files in os.walk(base):
                if "venv" in root or "work" in root.split(os.sep): continue
                for fn in files:
                    if fn == ".DS_Store": continue
                    p = os.path.join(root, fn)
                    z.write(p, os.path.relpath(p, ROOT)); n += 1
    print(f"{zname}: {os.path.getsize(zp)/1e9:.1f} GB, {n} files", flush=True)

for zname, dirs in SETS.items():
    build(zname, dirs)

with open(f"{DIST}/MANIFEST.md", "w") as m:
    m.write("# Bhote Koshi 2026 — full data release manifest\n\n"
            "| archive | size | sha256 |\n|---|---|---|\n")
    for fn in sorted(os.listdir(DIST)):
        if not fn.endswith(".zip"): continue
        p = f"{DIST}/{fn}"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        m.write(f"| {fn} | {os.path.getsize(p)/1e9:.2f} GB | {h.hexdigest()} |\n")
    m.write("\nImagery (c) Planet Labs PBC / (c) Vantor, open data programs, CC BY-NC 4.0.\n"
            "Terrain: NASA NSIDC HMA DEM, Copernicus GLO-30, Copernicus Sentinel-2.\n"
            "Derived products and code: see bhotekoshi-2026-flood-data.zip and the repository.\n")
print("manifest written", flush=True)
