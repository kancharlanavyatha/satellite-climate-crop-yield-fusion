"""Sanity-check the .npy tiles produced by clean_sentinel_tiles.py.

Checks every tile for shape/dtype/NaN problems and blank (all-zero) tiles,
then saves a handful of visual previews -- a true-color composite and an
NDVI heatmap -- so you can actually look at a sample and confirm they
resemble real satellite imagery of a district, not noise.

Run from the workspace root after clean_sentinel_tiles.py:
    python sic/inspect_tiles.py

Requires: numpy, pandas, matplotlib
    pip install matplotlib --break-system-packages
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST_CSV = ROOT / "data" / "processed" / "tile_manifest.csv"
PREVIEW_DIR = ROOT / "data" / "processed" / "tile_previews"

EXPECTED_SHAPE = (224, 224, 6)
BAND_NAMES = ["B2", "B3", "B4", "B8", "NDVI", "NDWI"]

# Sentinel-2 SR reflectance bands are scaled ~0-10000 (not 0-1). Anything far
# outside this is a sign something went wrong upstream (wrong band order,
# unscaled data, etc). NDVI/NDWI are normalized differences and must be in
# [-1, 1] by definition -- any value outside that range is a bug, not noise.
REFLECTANCE_MAX_PLAUSIBLE = 15000
N_PREVIEWS = 6  # how many tiles to render as images, spread across the manifest


def check_tile(path: Path) -> dict:
    issues = []
    arr = np.load(path)

    if arr.shape != EXPECTED_SHAPE:
        issues.append(f"shape is {arr.shape}, expected {EXPECTED_SHAPE}")
    if arr.dtype != np.float32:
        issues.append(f"dtype is {arr.dtype}, expected float32")
    if np.isnan(arr).any():
        issues.append("contains NaN")
    if np.isinf(arr).any():
        issues.append("contains Inf")
    if np.all(arr == 0):
        issues.append("entirely zero (fully blank tile)")

    if arr.ndim == 3 and arr.shape[-1] == 6:
        for i, name in enumerate(BAND_NAMES):
            band = arr[..., i]
            if name in ("NDVI", "NDWI"):
                if band.min() < -1.0001 or band.max() > 1.0001:
                    issues.append(f"{name} out of [-1,1] range: [{band.min():.3f}, {band.max():.3f}]")
            else:
                if band.max() > REFLECTANCE_MAX_PLAUSIBLE:
                    issues.append(f"{name} implausibly high: max={band.max():.0f}")
                if band.min() < 0:
                    issues.append(f"{name} has negative reflectance: min={band.min():.0f}")

    return {"issues": issues, "array": arr}


def save_preview(district: str, year: int, arr: np.ndarray, out_dir: Path) -> None:
    # Bands: 0=B2(blue) 1=B3(green) 2=B4(red) 3=B8(NIR) 4=NDVI 5=NDWI
    rgb = arr[..., [2, 1, 0]]  # true color = R, G, B
    # Reflectance bands aren't 0-1, so stretch to the tile's own range for display
    denom = (rgb.max() - rgb.min()) or 1.0
    rgb_display = np.clip((rgb - rgb.min()) / denom, 0, 1)
    ndvi = arr[..., 4]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb_display)
    axes[0].set_title(f"{district} {year} - true color")
    axes[0].axis("off")
    im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("NDVI")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{district}_{year}_preview.png", bbox_inches="tight", dpi=100)
    plt.close(fig)


def main() -> None:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Missing required file: {MANIFEST_CSV}")

    manifest = pd.read_csv(MANIFEST_CSV)
    problem_rows = []
    checked = 0

    for _, row in manifest.iterrows():
        tile_path = ROOT / row["tile_path"]
        if not tile_path.exists():
            problem_rows.append((row["District"], row["Year"], ["file missing on disk"]))
            continue
        result = check_tile(tile_path)
        checked += 1
        if result["issues"]:
            problem_rows.append((row["District"], row["Year"], result["issues"]))

    print(f"Checked {checked}/{len(manifest)} tiles")
    if problem_rows:
        print(f"\n{len(problem_rows)} tile(s) with issues:")
        for district, year, issues in problem_rows:
            print(f"  - {district} {year}: {'; '.join(issues)}")
    else:
        print("No structural or range issues found in any tile.")

    # Preview the worst- and best-quality tiles by nodata fraction, rather than
    # a random spread -- this is what actually tells you whether your poorest
    # rows (like the one you're worried about) are salvageable or should be
    # dropped, and gives you a real best-case tile to compare against.
    n_each = max(1, N_PREVIEWS // 2)
    ranked = manifest.sort_values("tile_nodata_fraction")
    sample = pd.concat([ranked.head(n_each), ranked.tail(n_each)]).drop_duplicates()
    print(f"\nPreviewing the {n_each} cleanest and {n_each} noisiest tiles by tile_nodata_fraction.")
    for _, row in sample.iterrows():
        tile_path = ROOT / row["tile_path"]
        if not tile_path.exists():
            continue
        arr = np.load(tile_path)
        if arr.shape == EXPECTED_SHAPE:
            save_preview(row["District"], row["Year"], arr, PREVIEW_DIR)

    print(f"\nSaved {len(sample)} preview image(s) to: {PREVIEW_DIR}")
    print("Open those PNGs -- the left panel should look like a real aerial/"
          "satellite photo of farmland (greens/browns, field patterns), and "
          "the right NDVI panel should show green over vegetated cropland and "
          "red/yellow over bare soil or water, not a uniform flat color.")


if __name__ == "__main__":
    main()