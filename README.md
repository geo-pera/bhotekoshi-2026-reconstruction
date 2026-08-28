# Bhote Koshi 2026 Outburst Flood — Reconstruction

Quantitative reconstruction of the 26 August 2026 Bhote Koshi–Trishuli outburst
flood (Rasuwa District, Nepal / Gyirong County, TAR China) from open satellite
data: trigger verification, flow-height and velocity measurements, sediment
deposition mapping, and calibrated 1D/2D flood simulation.

**This is a rapid scientific reconstruction, not an official assessment.**
Findings carry the uncertainties documented in `docs/`, and casualty figures
referenced anywhere are reported values, not products of this analysis.

## Repository layout
```
sim/scripts/     analysis + simulation code (python)
sim/inputs/      derived measurements (centerlines, trimlines, velocities, dh)
sim/dem/         terrain products (large; rebuilt by pipeline, not committed)
sim/masks/       cloud/water/disturbance rasters (rebuilt by pipeline)
sim/runs/        simulation outputs (rebuilt by pipeline)
docs/            co-registration audit, observation table
tools/           packaging utilities
```

## Data acquisition (all public)
```bash
# Planet Crisis Response (PlanetScope + SkySat + Pelican), CC BY-NC 4.0
aws s3 sync --no-sign-request s3://us-west-2.opendata.source.coop/planet/disasterdata/nepal-flash-flood-2026-08-26/ nepal-flash-flood-2026-08-26/
# Vantor Open Data (WorldView), CC BY-NC 4.0
aws s3 sync --no-sign-request s3://vantor-opendata/events/Nepal-Flooding-Aug-2026/ vantor/
# Copernicus GLO-30 tiles N27-N28 / E084-E085
aws s3 cp --no-sign-request s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N28_00_E085_00_DEM/Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif terrain/copernicus-glo30/   # (and N27/E084 variants)
# NASA High Mountain Asia 8 m DEM mosaic tiles 642,643,675,676 — requires free
# NASA Earthdata login: https://nsidc.org/data/hma_dem8m_mos  -> terrain/hma-8m/
```

## Pipeline (order matters)
1. `xsections.py` after building the DEM (datum-corrected, void-filled — see docs/COREGISTRATION.md)
2. `run_masks.py` — OmniCloudMask/OmniWaterMask per scene
3. `disturbance.py` → `trimline.py` → `superelevation.py` — measurements
4. `stereo_dh_dense.py` — deposition from WV03 opposite-look parallax
   (requires a user-supplied dense tie-point module `tie_points.py`; the
   required interface is documented in the script header — AROSICS or any
   phase-correlation tie-point engine can back it)
5. `route1d.py` — calibrated 1D routing; `swe2d.py` / `swe2d_torch.py` — 2D
6. `export_viewer2.py` / `animate*.py` — visualization products

## Environment
Python ≥ 3.11 with: numpy, gdal (osgeo), rasterio, torch, opencv-python,
matplotlib, pillow, omnicloudmask, omniwatermask.

## Licenses
Code: MIT (see LICENSE). Derived data products inherit **CC BY-NC 4.0** from the
source imagery (© Planet Labs PBC, © Vantor — open data programs). Terrain
derived from NASA NSIDC HMA DEM and Copernicus GLO-30 (free use with attribution).
