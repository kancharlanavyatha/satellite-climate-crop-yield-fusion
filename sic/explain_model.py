"""Feature Importance & Modality Attribution Analysis for Crop Yield Prediction.

Analyzes the relative importance of:
- Satellite Spectral Indices (NDVI, NDWI, EVI, SAVI, GCVI)
- Climate & Weather Features (Rainfall, Temperature, Seasonality, Heat Stress)
- Soil Properties (pH, Clay, Sand, Organic Carbon, Slope, Elevation)
- Agricultural Management (Sown Area)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "processed" / "multimodal_features_extracted.csv"
IMPORTANCE_JSON = ROOT / "data" / "processed" / "feature_importance.json"
PLOT_PNG = ROOT / "data" / "processed" / "feature_importance_plot.png"

TABULAR_COLS = [
    "SownArea", "HeatStressDays_gt35C", "Rain_Early_mm", "Rain_Late_mm",
    "Rain_Mid_mm", "Rain_Total_mm", "Temp_Mean_C", "Clay_pct", "Elevation_m",
    "OrganicCarbon_g_kg", "Sand_pct", "Slope_deg", "Soil_pH",
    "Rain_Early_Share", "Rain_Mid_Share", "Rain_Late_Share",
    "Rain_GrowthPhase_mm", "Rain_Seasonality_Index", "Heat_Rain_Stress"
]

SATELLITE_COLS = [
    "valid_pixel_ratio", "ndvi_mean", "ndvi_std", "ndvi_median", "ndvi_p10",
    "ndvi_p90", "ndwi_mean", "ndwi_std", "ndwi_median", "nir_mean", "nir_std",
    "red_mean", "green_mean", "blue_mean", "evi_mean", "evi_std", "savi_mean",
    "gcvi_mean"
]

ALL_FEATURES = TABULAR_COLS + SATELLITE_COLS


def get_feature_category(col_name: str) -> str:
    if col_name in ["SownArea"]:
        return "Agriculture / Management"
    elif "Rain" in col_name or "Temp" in col_name or "Heat" in col_name:
        return "Weather & Climate Stress"
    elif col_name in ["Clay_pct", "Sand_pct", "OrganicCarbon_g_kg", "Soil_pH", "Slope_deg", "Elevation_m"]:
        return "Soil & Terrain"
    else:
        return "Satellite Spectral Indices"


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data file missing: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    X = df[ALL_FEATURES].values
    y = df["Yield_Tonne_per_Hectare"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ExtraTrees & RandomForest on all features for robust feature ranking
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_scaled, y)

    et = ExtraTreesRegressor(n_estimators=200, random_state=42)
    et.fit(X_scaled, y)

    avg_importances = (rf.feature_importances_ + et.feature_importances_) / 2.0

    importance_df = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Importance": avg_importances,
        "Category": [get_feature_category(c) for c in ALL_FEATURES]
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Calculate Modality Attribution Shares
    category_shares = importance_df.groupby("Category")["Importance"].sum().to_dict()

    print("--- Modality Importance Shares ---")
    for cat, share in sorted(category_shares.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<30}: {share * 100:.2f}%")

    print("\n--- Top 10 Most Influential Features ---")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['Feature']:<25} ({row['Category']}): {row['Importance']*100:.2f}%")

    IMPORTANCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    out_dict = {
        "top_features": importance_df.to_dict(orient="records"),
        "modality_shares": category_shares
    }
    with open(str(IMPORTANCE_JSON), "w") as f:
        json.dump(out_dict, f, indent=2)

    # Generate Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    top15 = importance_df.head(15).iloc[::-1] # Reverse for horizontal bar chart
    colors = {
        "Satellite Spectral Indices": "#2ca02c",
        "Weather & Climate Stress": "#1f77b4",
        "Soil & Terrain": "#8c564b",
        "Agriculture / Management": "#ff7f0e"
    }
    bar_colors = [colors[cat] for cat in top15["Category"]]

    plt.barh(top15["Feature"], top15["Importance"] * 100, color=bar_colors)
    plt.xlabel("Importance Share (%)")
    plt.title("Top 15 Feature Importances for Crop Yield Prediction")
    plt.tight_layout()
    plt.savefig(str(PLOT_PNG), dpi=150)
    plt.close()

    print(f"\nFeature importance analysis complete! Saved JSON to {IMPORTANCE_JSON} and plot to {PLOT_PNG}")


if __name__ == "__main__":
    main()
