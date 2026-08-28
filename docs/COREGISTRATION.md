# Co-registration audit

All raster products are analyzed on common grids in EPSG:32645 (WGS84 / UTM 45N),
warped once at ingest. Vertical datum: EGM2008 orthometric (HMA ellipsoidal
heights corrected by a robust block-median fitted offset surface, residual σ 1.4 m).

## Measured alignment (median shifts, tie-point phase correlation)

| Pair | Shift | Note |
|---|---|---|
| HMA 8 m DEM ↔ Copernicus GLO-30 | 5.5 m | Nuth–Kääb slope/aspect regression; sub-pixel at 30 m |
| HMA ↔ GLO-30 vertical (pre-fix) | −35.5 m | ellipsoid/geoid offset; corrected (see above) |
| PlanetScope pre ↔ post | ~0 m | phase correlation, stable chips |
| DEM centerline ↔ Planet imagery | ≤ 1–2 px @30 m | visual overlay verification |
| WV02-2021 ↔ WV03-2026 (Vantor) | 0.7 m global | spread 3.7 m (ortho terrain distortion) |
| WV02-2021 ↔ PlanetScope pre | 3.3 m | < 1 Planet pixel |
| WV03 ↔ SkySat (post pair) | < 1 m median | few ties (cloud) |

## Local registration of presentation pairs

Pre/post comparison images (before/after pairs) were additionally locally
co-registered (dense tie-point shift field, cubic remap of the post image onto
the 2021 baseline). Residuals after registration, checked with independent ties:

| Window | Median residual | Checks |
|---|---|---|
| Syabrubesi | 0.50 m | 11 |
| Rasuwagadhi border | 1.75 m | 11 |
| Timure | 2.10 m | 4 |

Residuals are limited by surviving common texture: in heavily destroyed areas
only slope features constrain the match. Analysis rasters (8 m grid) are
unaffected at these magnitudes.
