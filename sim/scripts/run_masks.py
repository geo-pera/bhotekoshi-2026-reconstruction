"""Cloud + water masks for Planet scenes using OmniCloudMask / OmniWaterMask.

Outputs into sim/masks/:
  <id>_cloud.tif      uint8 OCM classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 shadow
  <id>_water_raw.tif  OWM output (written by make_water_mask)
  <id>_water.tif      uint8: 1 water observed clear-sky, 0 land, 255 cloud/invalid
PlanetScope analytic band order in file: 1=B 2=G 3=R 4=N.
"""
import sys, glob, os, shutil
from pathlib import Path
import numpy as np
import rasterio
from omnicloudmask import predict_from_array as ocm_predict
from omniwatermask import make_water_mask

ROOT = "/nepal-flash-flood-2026-08-26"
OUT = Path("/sim/masks")
OUT.mkdir(parents=True, exist_ok=True)

SCENES = sys.argv[1:] or [
    "20260826_050125_99_255f",   # post, clearest (14%)
    "20260826_050135_34_255f",   # post, 11% clear
    "20260826_054456_67_251f",   # post, covers Lende Khola + source
    "20260527_053226_41_254a",   # pre, near-clear baseline
    "20260527_053219_95_254a",   # pre, covers lower Lende Khola
]

for sid in SCENES:
    hits = glob.glob(f"{ROOT}/*/*/items/{sid}/{sid}_analytic*.tif")
    if not hits:
        print(f"!! no analytic asset for {sid}", flush=True); continue
    path = hits[0]
    print(f"== {sid}", flush=True)
    with rasterio.open(path) as src:
        g, r, n = (src.read(i).astype(np.float32) for i in (2, 3, 4))
        profile = src.profile
    valid = (g + r + n) > 0

    cloud = ocm_predict(np.stack([r, g, n]), inference_device=None)
    cloud = np.asarray(cloud).squeeze().astype(np.uint8)
    cloud[~valid] = 255
    p = profile.copy(); p.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(OUT / f"{sid}_cloud.tif", "w", **p) as dst:
        dst.write(cloud, 1)
    print(f"   cloud: {(cloud[valid]==0).mean()*100:.1f}% clear in footprint", flush=True)
    del g, r, n

    if os.environ.get("SKIP_WATER"):
        print("   (water mask skipped)", flush=True); continue
    # OWM runs from file; band_order = [R, G, B, NIR] as 1-indexed bands -> BGRN file = [3,2,1,4]
    res = make_water_mask(scene_paths=[path], band_order=[3, 2, 1, 4],
                          output_dir=OUT, use_osm_building=False, use_osm_roads=False)
    raw_path = OUT / f"{sid}_water_raw.tif"
    shutil.move(str(res[0]), raw_path)
    with rasterio.open(raw_path) as src:
        water = src.read(1).astype(np.uint8)
    water[cloud != 0] = 255
    with rasterio.open(OUT / f"{sid}_water.tif", "w", **p) as dst:
        dst.write(water, 1)
    ok = valid & (cloud == 0)
    print(f"   water: {(water[ok]==1).mean()*100:.1f}% of clear pixels" if ok.any() else "   no clear pixels", flush=True)
print("all done", flush=True)
