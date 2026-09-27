"""Validate raw Sentinel-2 GeoTIFF exports and convert them to model-ready
.npy tiles. This is the ONLY script that reads data/raw/sentinel_tiff/ -- once
it has run, the TIFFs can be deleted; everything downstream uses the manifest
CSV and the .npy files it produces here.

Run from the workspace root after downloading all Earth Engine tile exports:
    python sic/clean_sentinel_tiles.py

Requires: numpy, pandas, rasterio, opencv-python
    pip install rasterio opencv-python --break-system-packages
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent
TIFF_DIR = ROOT / "data" / "raw" / "sentinel_tiff"
TILES_DIR = ROOT / "data" / "processed" / "tiles"
MANIFEST_CSV = ROOT / "data" / "processed" / "tile_manifest.csv"

TILE_SIZE = 224
EXPECTED_BANDS = 6
BAND_NAMES = ["B2", "B3", "B4", "B8", "NDVI", "NDWI"]

# 03_export_sentinel_tiles.js clips each export to district.geometry(), but
# Export.image.toDrive rasterizes the *bounding box* of that geometry, so a
# large, CONSTANT share of every tile is legitimately outside the district
# polygon and comes back as NaN every year -- that's shape padding, not cloud
# cover. Only reject a tile if it is essentially all nodata (a corrupt/empty
# export); real cloud-heavy years are kept but flagged via nodata_fraction in
# the manifest so they can be filtered downstream if desired.
MAX_NODATA_FRACTION = 0.98

FILENAME_RE = re.compile(r"^(?P<district>.+)_(?P<year>\d{4})_Kharif\.tiff?$")

# 03_export_sentinel_tiles.js's own alias map normalizes most historical GAUL
# spellings (Ananthapur, Cuddapah, etc.) to the canonical name before the
# filename is built -- except Vishakhapatnam, which it never added. Fold that
# one remaining case here so the manifest lines up with the canonical
# district names used in engineered_tabular_features.csv.
RAW_TO_CANONICAL_DISTRICT = {
    "Vishakhapatnam": "Visakhapatnam",
}


def parse_filename(path: Path) -> tuple[str, int] | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    raw_district = match.group("district").replace("_", " ")
    district = RAW_TO_CANONICAL_DISTRICT.get(raw_district, raw_district)
    return district, int(match.group("year"))


def load_and_validate(path: Path) -> tuple[np.ndarray | None, str | None, float]:
    """Return (array shaped H x W x 6, reason-if-rejected, nodata_fraction)."""
    with rasterio.open(path) as src:
        if src.count != EXPECTED_BANDS:
            return None, f"expected {EXPECTED_BANDS} bands, found {src.count}", 1.0
        arr = src.read().astype("float32")  # shape: (bands, H, W)

    arr = np.moveaxis(arr, 0, -1)  # -> (H, W, bands)
    nodata_fraction = float(np.isnan(arr).mean())
    if nodata_fraction > MAX_NODATA_FRACTION:
        return None, f"{nodata_fraction:.1%} NaN -- essentially empty export", nodata_fraction

    if nodata_fraction > 0:
        # NaN here is mostly bounding-box padding outside the district's
        # actual polygon shape (see MAX_NODATA_FRACTION comment above), so
        # fill with 0 -- "no signal / background" -- rather than the band
        # mean, which would fabricate plausible-looking reflectance there.
        arr = np.nan_to_num(arr, nan=0.0)

    return arr, None, nodata_fraction


def resize_tile(arr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(arr, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
    return resized.astype("float32")


def main() -> None:
    if not TIFF_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {TIFF_DIR}")

    tiff_paths = sorted(list(TIFF_DIR.glob("*.tif")) + list(TIFF_DIR.glob("*.tiff")))
    if not tiff_paths:
        raise FileNotFoundError(f"No .tif/.tiff files found in {TIFF_DIR}")

    TILES_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped: list[dict] = []

    for tiff_path in tiff_paths:
        parsed = parse_filename(tiff_path)
        if parsed is None:
            skipped.append({"file": tiff_path.name, "reason": "filename doesn't match <District>_<Year>_Kharif.tif"})
            continue
        district, year = parsed

        arr, error, nodata_fraction = load_and_validate(tiff_path)
        if error is not None:
            skipped.append({"file": tiff_path.name, "reason": error})
            continue

        tile = resize_tile(arr)
        out_name = f"{re.sub(r'[^A-Za-z0-9]', '_', district)}_{year}.npy"
        out_path = TILES_DIR / out_name
        np.save(out_path, tile)

        records.append({
            "District": district,
            "Year": year,
            "tile_path": str(out_path.relative_to(ROOT)),
            "tile_nodata_fraction": round(nodata_fraction, 4),
        })

    manifest = pd.DataFrame.from_records(records).sort_values(["Year", "District"]).reset_index(drop=True)
    duplicates = manifest.duplicated(["District", "Year"])
    if duplicates.any():
        dup_rows = manifest.loc[duplicates, ["District", "Year"]].to_dict("records")
        raise ValueError(f"Duplicate district-year tiles found: {dup_rows}")

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_CSV, index=False)

    print(f"Bands (fixed order in every tile): {BAND_NAMES}")
    print(f"Created: {MANIFEST_CSV}")
    print(f"Tiles written: {len(manifest)}/{len(tiff_paths)} -> {TILES_DIR}")
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped:")
        for item in skipped:
            print(f"  - {item['file']}: {item['reason']}")
    print("\nOnce you've confirmed the manifest and tiles look right, the raw "
          "TIFFs in data/raw/sentinel_tiff/ are safe to delete -- nothing "
          "downstream reads them again.")


if __name__ == "__main__":
    main()