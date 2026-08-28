import sys
import numpy as np
import rasterio
from omnicloudmask import predict_from_array as ocm
from omniwatermask import make_water_mask
from pathlib import Path
import shutil

import os as _os
ROOT = _os.environ.get("BK_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
for date in ("20260824", "20260827"):
    path = f"{ROOT}/s2/{date}/RUL_bgrn.tif"
    with rasterio.open(path) as src:
        b, g, r, n = (src.read(i).astype(np.float32) for i in (1, 2, 3, 4))
        prof = src.profile
    valid = (b + g + r + n) > 0
    cloud = np.asarray(ocm(np.stack([r, g, n]))).squeeze().astype(np.uint8)
    cloud[~valid] = 255
    p = prof.copy(); p.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(f"{ROOT}/s2/{date}/RUL_cloud.tif", "w", **p) as dst:
        dst.write(cloud, 1)
    print(f"{date}: {(cloud[valid]==0).mean()*100:.1f}% clear", flush=True)
    res = make_water_mask(scene_paths=[path], band_order=[3, 2, 1, 4],
                          output_dir=Path(f"{ROOT}/s2/{date}"),
                          use_osm_building=False, use_osm_roads=False)
    shutil.move(str(res[0]), f"{ROOT}/s2/{date}/RUL_water_raw.tif")
    with rasterio.open(f"{ROOT}/s2/{date}/RUL_water_raw.tif") as src:
        w = src.read(1).astype(np.uint8)
    w[cloud != 0] = 255
    with rasterio.open(f"{ROOT}/s2/{date}/RUL_water.tif", "w", **p) as dst:
        dst.write(w, 1)
    ok = valid & (cloud == 0)
    print(f"{date}: water {(w[ok]==1).mean()*100:.2f}% of clear", flush=True)
print("done")
