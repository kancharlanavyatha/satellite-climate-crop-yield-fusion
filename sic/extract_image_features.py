"""Extract image statistics and vegetation indices from Sentinel-2 .npy tiles.

Computes mean, std, percentiles for NDVI, NDWI, EVI, SAVI, GCVI, and raw bands
over non-background (valid) pixels for each district-year tile.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FINAL_CSV = ROOT / "data" / "processed" / "final_features_with_tiles.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "multimodal_features_extracted.csv"


def process_tile(tile_path: Path) -> dict:
    arr = np.load(tile_path)  # Shape: (224, 224, 6)
    # Channels: 0:B2 (Blue), 1:B3 (Green), 2:B4 (Red), 3:B8 (NIR), 4:NDVI, 5:NDWI
    b2 = arr[..., 0]
    b3 = arr[..., 1]
    b4 = arr[..., 2]
    b8 = arr[..., 3]
    ndvi = arr[..., 4]
    ndwi = arr[..., 5]

    # Valid mask: pixels where NIR > 0 (non-background padding)
    valid_mask = b8 > 0
    valid_count = np.sum(valid_mask)
    total_count = arr.shape[0] * arr.shape[1]
    valid_ratio = valid_count / total_count if total_count > 0 else 0.0

    if valid_count == 0:
        # Fallback if entire tile is zero
        valid_mask = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)

    v_b2 = b2[valid_mask]
    v_b3 = b3[valid_mask]
    v_b4 = b4[valid_mask]
    v_b8 = b8[valid_mask]
    v_ndvi = ndvi[valid_mask]
    v_ndwi = ndwi[valid_mask]

    # Derived indices
    # Scaled reflectance (0-10000 to 0-1 range for index calculations)
    r = np.clip(v_b4 / 10000.0, 0, 1)
    g = np.clip(v_b3 / 10000.0, 0, 1)
    b = np.clip(v_b2 / 10000.0, 0, 1)
    nir = np.clip(v_b8 / 10000.0, 0, 1)

    evi = 2.5 * (nir - r) / (nir + 6.0 * r - 7.5 * b + 1.0 + 1e-6)
    savi = 1.5 * (nir - r) / (nir + r + 0.5 + 1e-6)
    gcvi = (nir / (g + 1e-6)) - 1.0

    stats = {
        "valid_pixel_ratio": valid_ratio,
        "ndvi_mean": float(np.mean(v_ndvi)),
        "ndvi_std": float(np.std(v_ndvi)),
        "ndvi_median": float(np.median(v_ndvi)),
        "ndvi_p10": float(np.percentile(v_ndvi, 10)),
        "ndvi_p90": float(np.percentile(v_ndvi, 90)),
        "ndwi_mean": float(np.mean(v_ndwi)),
        "ndwi_std": float(np.std(v_ndwi)),
        "ndwi_median": float(np.median(v_ndwi)),
        "nir_mean": float(np.mean(v_b8)),
        "nir_std": float(np.std(v_b8)),
        "red_mean": float(np.mean(v_b4)),
        "green_mean": float(np.mean(v_b3)),
        "blue_mean": float(np.mean(v_b2)),
        "evi_mean": float(np.mean(evi)),
        "evi_std": float(np.std(evi)),
        "savi_mean": float(np.mean(savi)),
        "gcvi_mean": float(np.mean(gcvi)),
    }
    return stats


def main():
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"Missing file: {FINAL_CSV}")

    df = pd.read_csv(FINAL_CSV)
    extracted_rows = []

    for idx, row in df.iterrows():
        tile_file = ROOT / row["tile_path"]
        if not tile_file.exists():
            raise FileNotFoundError(f"Tile file not found: {tile_file}")
        
        stats = process_tile(tile_file)
        full_row = row.to_dict()
        full_row.update(stats)
        extracted_rows.append(full_row)

    out_df = pd.DataFrame(extracted_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Extracted image features for {len(out_df)} rows. Saved to: {OUTPUT_CSV}")
    print("New image features added:", [col for col in out_df.columns if col not in df.columns])


if __name__ == "__main__":
    main()
