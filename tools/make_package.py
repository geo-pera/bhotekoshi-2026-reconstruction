"""Build the public data package (dist/bhotekoshi-2026-flood-data.zip)."""
import os, shutil, zipfile, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = f"{ROOT}/dist/bhotekoshi-2026-flood-data"
shutil.rmtree(PKG, ignore_errors=True)
for d in ("dem", "measurements", "masks", "model", "scenario", "viewer"):
    os.makedirs(f"{PKG}/{d}", exist_ok=True)

def cp(src, dst_dir):
    if os.path.exists(src):
        shutil.copy2(src, f"{PKG}/{dst_dir}/"); return True
    print("  missing:", src); return False

# terrain
cp(f"{ROOT}/sim/dem/domain_8m_filled_ortho.tif", "dem")
cp(f"{ROOT}/sim/dem/dh_surface_32m.tif", "dem")
# measurements
for f in ("trimline_profile.csv", "trimline_profile_v2.csv", "superelevation_velocities.csv",
          "stereo_dh.csv", "stereo_dh_dense.csv", "stereo_dh_dense_relaxed.csv",
          "rasuwa_bare_edge.csv", "centerline_v3.csv"):
    cp(f"{ROOT}/sim/inputs/{f}", "measurements")
# masks
cp(f"{ROOT}/sim/masks/disturbance_v2.tif", "masks")
cp(f"{ROOT}/sim/masks/disturbance.tif", "masks")
cp(f"{ROOT}/s2/change_south.tif", "masks")
# model: cross-sections + key runs -> peak GeoTIFFs
for f in glob.glob(f"{ROOT}/sim/inputs/xsections*.npz"):
    shutil.copy2(f, f"{PKG}/model/")
cp(f"{ROOT}/sim/inputs/lende_burn_profile.npy", "model")
cp(f"{ROOT}/sim/runs/sweep.log", "model")

def npz_to_tifs(npz_path, prefix, outdir):
    from osgeo import gdal, osr
    gdal.UseExceptions()
    if not os.path.exists(npz_path): print("  missing:", npz_path); return
    d = np.load(npz_path)
    gt = tuple(d["gt"]); Z = d["Z"]
    sr = osr.SpatialReference(); sr.ImportFromEPSG(32645)
    for key in ("peak_depth", "peak_eta", "peak_stage"):
        if key not in d.files: continue
        a = d[key].astype(np.float32)
        if a.ndim != 2: continue
        drv = gdal.GetDriverByName("GTiff")
        o = drv.Create(f"{PKG}/{outdir}/{prefix}_{key}.tif", a.shape[1], a.shape[0], 1,
                       gdal.GDT_Float32, ["COMPRESS=DEFLATE", "TILED=YES"])
        o.SetGeoTransform(gt); o.SetProjection(sr.ExportToWkt())
        o.GetRasterBand(1).WriteArray(np.where(np.isfinite(a), a, -9999))
        o.GetRasterBand(1).SetNoDataValue(-9999); o.FlushCache()
npz_to_tifs(f"{ROOT}/sim/runs/swe2d_gorge_8m_postDEM.npz", "event_2d_8m", "model")
npz_to_tifs(f"{ROOT}/sim/runs/swe2d_lende_breach_gpu2.npz", "breach_scenario_5Mm3", "scenario")
cp(f"{ROOT}/sim/runs/flood_viewer_standalone.html", "viewer")
# docs
shutil.copy2(f"{ROOT}/docs/COREGISTRATION.md", PKG)
shutil.copy2(f"{ROOT}/docs/OBSERVATIONS.csv", PKG)
readme = f"""# Bhote Koshi 2026 Outburst Flood — Open Data Package

Derived data from a rapid reconstruction of the 26 Aug 2026 Bhote Koshi–Trishuli
outburst flood, produced entirely from openly licensed satellite data within
~72 hours of the event. NOT an official assessment; see caveats below.

## Contents
- dem/            8 m terrain (EGM2008-corrected, void-filled HMA); stereo-parallax
                  elevation-change surface (32 m; deposition wedge km 33-38)
- measurements/   trimline flow heights (217 banks), superelevation velocities,
                  deposition point measurements, river centerline w/ chainage
- masks/          flood/debris disturbance rasters (8 m)
- model/          channel cross-sections, calibration sweep, 2D peak-depth/stage GeoTIFFs
- scenario/       HYPOTHETICAL barrier-lake breach (5 Mm3) peak fields — indicative only
- viewer/         self-contained interactive 3D model (open in a browser)
- OBSERVATIONS.csv  every observational constraint used, with sources
- COREGISTRATION.md alignment audit for all layers

CRS: EPSG:32645 (WGS84/UTM 45N). Vertical: EGM2008. NoData: -9999.

## Key caveats
- Trimlines record PEAK STAGE (runup/splash included); >110 m values may include
  flood-triggered slope failures. Chainage 30.5-33 km excluded (haze artifact).
- Deposition is a lower bound; smooth fresh mud defeats image correlation.
- The scenario is model output on partially artifact-corrected terrain,
  clearly hypothetical; defer to NDRRMA for any safety decision.
- Gauge figures are as publicly reported, not raw agency records.

## License & attribution
Derived from imagery (c) Planet Labs PBC and (c) Vantor via their open data
programs: this package inherits CC BY-NC 4.0 (non-commercial, attribution).
Terrain inputs: NASA NSIDC High Mountain Asia 8 m DEM; Copernicus GLO-30.
Building/road references: (c) OpenStreetMap contributors (ODbL).
Produced by geopera, Aug 2026.
"""
open(f"{PKG}/README.md", "w").write(readme)
zp = f"{ROOT}/dist/bhotekoshi-2026-flood-data.zip"
if os.path.exists(zp): os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(PKG):
        for fn in files:
            p = os.path.join(base, fn)
            z.write(p, os.path.relpath(p, os.path.dirname(PKG)))
print(f"package: {os.path.getsize(zp)/1048576:.0f} MB -> {zp}")
