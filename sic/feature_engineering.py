"""Merge cleaned labels, weather, soil, and terrain into the team's tabular input.

Run from the workspace root after downloading the two Earth Engine CSV exports:
    python sic/feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "data" / "processed" / "labels_ap_kharif_rice_2017_2021.csv"
STATIC = ROOT / "data" / "raw" / "features" / "district_static_features.csv"
WEATHER = ROOT / "data" / "raw" / "features" / "district_year_weather.csv"
OUTPUT = ROOT / "data" / "processed" / "engineered_tabular_features.csv"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    return df.drop(columns=[column for column in ["system:index", ".geo"] if column in df.columns])


def normalize_district(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Ananthapur": "Anantapur",
        "Cuddapah": "Kadapa",
        "Y.S.R. Kadapa": "Kadapa",
        "Sri Potti Sriramulu Nellore": "Nellore",
        "Vishakhapatnam": "Visakhapatnam",
    }
    df = df.copy()
    df["District"] = df["District"].astype(str).str.strip().replace(aliases)
    return df


def main() -> None:
    labels = normalize_district(load_csv(LABELS))
    static = normalize_district(load_csv(STATIC))
    weather = normalize_district(load_csv(WEATHER))
    weather["Year"] = pd.to_numeric(weather["Year"], errors="raise").astype(int)

    if static["District"].duplicated().any():
        raise ValueError("Static features must have one row per district.")
    if weather.duplicated(["District", "Year"]).any():
        raise ValueError("Weather features must have one row per district-year.")

    # Production and source-yield checks are intentionally excluded: production
    # is used to calculate the target and would leak the answer into the model.
    labels = labels[["District", "Year", "Season", "Crop", "SownArea", "Yield_Tonne_per_Hectare"]]
    weather = weather.drop(columns=[column for column in ["Season"] if column in weather.columns])

    merged = labels.merge(weather, on=["District", "Year"], how="left", validate="one_to_one")
    merged = merged.merge(static, on="District", how="left", validate="many_to_one")
    missing = merged.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(f"Missing merged values:\n{missing.to_string()}")

    # Our original climate-stress features. Scaling belongs to the model's
    # training pipeline, so these remain in real, documented units here.
    total = merged["Rain_Total_mm"].replace(0, np.nan)
    merged["Rain_Early_Share"] = merged["Rain_Early_mm"] / total
    merged["Rain_Mid_Share"] = merged["Rain_Mid_mm"] / total
    merged["Rain_Late_Share"] = merged["Rain_Late_mm"] / total
    merged["Rain_GrowthPhase_mm"] = merged["Rain_Early_mm"] + merged["Rain_Mid_mm"]
    merged["Rain_Seasonality_Index"] = (merged["Rain_Mid_mm"] - merged["Rain_Early_mm"]) / total
    merged["Heat_Rain_Stress"] = merged["HeatStressDays_gt35C"] / (merged["Rain_Total_mm"] + 1.0)

    if merged.isna().any().any():
        raise ValueError("Feature engineering created missing values.")
    merged = merged.sort_values(["Year", "District"]).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False, float_format="%.6f")

    print(f"Created: {OUTPUT}")
    print(f"Rows: {len(merged)} | districts: {merged['District'].nunique()} | years: {merged['Year'].min()}-{merged['Year'].max()}")
    print("Target: Yield_Tonne_per_Hectare")
    print("Do not use Production_Tonnes as a model feature; it is deliberately excluded.")


if __name__ == "__main__":
    main()
