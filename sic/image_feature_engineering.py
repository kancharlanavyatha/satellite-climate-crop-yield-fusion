"""Merge the image tile manifest into the tabular feature table.

This script never reads a TIFF -- it only needs tile_manifest.csv (written by
clean_sentinel_tiles.py) and engineered_tabular_features.csv (written by
feature_engineering.py). That means it still works after the raw Sentinel
TIFFs and even the .npy tiles have been archived elsewhere, as long as the
manifest and tile files are present.

Run from the workspace root after clean_sentinel_tiles.py has produced
tile_manifest.csv:
    python sic/image_feature_engineering.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TABULAR_CSV = ROOT / "data" / "processed" / "engineered_tabular_features.csv"
MANIFEST_CSV = ROOT / "data" / "processed" / "tile_manifest.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "final_features_with_tiles.csv"


def main() -> None:
    if not TABULAR_CSV.exists():
        raise FileNotFoundError(f"Missing required file: {TABULAR_CSV}")
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Missing required file: {MANIFEST_CSV}\n"
            "Run clean_sentinel_tiles.py first to generate it."
        )

    tabular = pd.read_csv(TABULAR_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)

    if manifest.duplicated(["District", "Year"]).any():
        raise ValueError("tile_manifest.csv has more than one tile for the same District-Year.")

    merged = tabular.merge(manifest, on=["District", "Year"], how="left")

    n_before = len(merged)
    missing = merged[merged["tile_path"].isna()][["District", "Year"]]
    final = merged.dropna(subset=["tile_path"]).reset_index(drop=True)
    n_after = len(final)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")

    print(f"Created: {OUTPUT_CSV}")
    print(f"Rows: {n_after}/{n_before} kept (had a matching tile)")
    if not missing.empty:
        print(f"\n{len(missing)} row(s) had no matching tile in the manifest, dropped:")
        for _, row in missing.iterrows():
            print(f"  - {row['District']} {row['Year']}")


if __name__ == "__main__":
    main()