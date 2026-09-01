# Bhote Koshi 2026 Outburst Flood — Reconstruction

Quantitative reconstruction of the 26 August 2026 Bhote Koshi–Trishuli outburst
flood (Rasuwa District, Nepal / Gyirong County, TAR China) from open satellite
data: trigger verification, flow-height and velocity measurements, sediment
deposition mapping, and calibrated 1D/2D flood simulation.

**This is a rapid scientific reconstruction, not an official assessment.**
Findings carry the uncertainties documented in `docs/`, and casualty figures
referenced anywhere are reported values, not products of this analysis.

## Correction (1 September 2026)

The ortho-parallax elevation-change product released earlier
(`deposition_wedge.geojson`, `erosion_zones.geojson`, and the sediment-budget
figures derived from it: a 10–18 m deposition wedge at corridor km 33–38 and
+19.7/−5.0 Mm³ of deposition/erosion) **is retracted**. The conversion from
image offsets to height assumed the two WorldView-3 looks view the valley from
opposite azimuths; the pair is in fact same-side in-track stereo, so the
conversion factor was wrong in both sign and magnitude, and the map was
dominated by registration residuals between the vendor orthos rather than by
elevation change. The error was caught by differencing a rigorous 0.5 m
photogrammetric DSM (built from the RPC-bearing stereo strips Vantor released
on 31 August) against the pre-event DEM, and confirmed by re-deriving the
viewing geometry from the strips' RPC cameras.

The superseding products — the DSM itself, elevation-change and uncertainty
rasters, corrected deposition/erosion polygons, a per-km sediment budget, and
a building-level damage census — are in `vectors/` and the release assets.
The measured story reverses the retracted one: the corridor above Syabrubesi
is erosional (2–12 m of floor lowering), the main deposit (12–18 m thick) sits
where the valley opens at km 40.5–43.6, and the river has re-incised 13–21 m
where it confines again downstream. Simulation results were not affected (the
flood models run on the pre-event DEM), and the barrier-lake breach scenario
does not touch the affected reach.

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
4. ~~`stereo_dh_dense.py` — deposition from WV03 ortho parallax~~ **retracted**
   (wrong viewing-geometry assumption; kept only for the record — see
   "Correction" above). Elevation change now comes from the 0.5 m stereo DSM
   differenced against the pre-event DEM (release assets `dsm_reach*_0p5m.tif`,
   `dh_reach*_2m.tif`, `sigma_reach*_2m.tif`)
5. `route1d.py` — calibrated 1D routing; `swe2d.py` / `swe2d_torch.py` — 2D
6. `export_viewer2.py` / `animate*.py` — visualization products

## Environment
Python ≥ 3.11 with: numpy, gdal (osgeo), rasterio, torch, opencv-python,
matplotlib, pillow, omnicloudmask, omniwatermask.

## Licenses
Code: MIT (see LICENSE). Derived data products inherit **CC BY-NC 4.0** from the
source imagery (© Planet Labs PBC, © Vantor — open data programs). Terrain
derived from NASA NSIDC HMA DEM and Copernicus GLO-30 (free use with attribution).

## Vector products (`vectors/`)
GIS-ready GeoJSON layers (EPSG:4326) plus a combined GeoPackage
(`bhotekoshi_analysis.gpkg`): modelled event and breach-scenario inundation
extents, ensemble inundation-probability classes, DSM-measured
deposition/erosion polygons (`deposition_dsm.geojson`, `erosion_dsm.geojson` —
these supersede the retracted parallax layers), the building damage census for
both stereo reaches (`building_census_reachA/B.geojson`), the per-km sediment
budget (`per_km_budget.csv`), Sentinel-1 change candidates (including the
detachment-zone cluster), 217 trimline flow-height points, superelevation
velocity points, per-settlement modelled arrival times, the flow-path
centerline with km markers, day-2 disturbance polygons for the lower Trishuli,
and OSM buildings in or within 50 m of the detected scour. Regenerate with
`tools/make_vectors.py`. Same CC BY-NC 4.0 terms as other derived products.
Census caveat: Timure sits under the stereo cloud mask (1% DSM coverage), so
the reach A census is blind exactly where day-2 imagery shows heavy damage —
treat its counts as a lower bound, not a total.
