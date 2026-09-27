"""Train Hybrid Multimodal Models combining CNN Latent Spatial Embeddings,
Spectral Vegetation Indices, and Tabular Environmental Features into GBDTs.

Evaluates XGBoost, LightGBM, Extra Trees, Random Forest, and a Weighted Stacking Ensemble.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "processed" / "multimodal_features_extracted.csv"
EMBEDDINGS_CSV = ROOT / "data" / "processed" / "cnn_tabular_embeddings.csv"
RESULTS_JSON = ROOT / "data" / "processed" / "hybrid_multimodal_results.json"
HYBRID_PREDS_CSV = ROOT / "data" / "processed" / "hybrid_oof_predictions.csv"

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(rmse, 4),
        "MSE": round(float(mse), 4),
        "R2": round(float(r2), 4),
        "MAPE_pct": round(mape, 2)
    }


def main():
    if not DATA_CSV.exists() or not EMBEDDINGS_CSV.exists():
        raise FileNotFoundError("Missing data CSV or embeddings CSV.")

    df_data = pd.read_csv(DATA_CSV)
    df_emb = pd.read_csv(EMBEDDINGS_CSV)

    merged = df_data.merge(df_emb, on=["District", "Year"], how="inner")
    print(f"Merged dataset shape for hybrid modeling: {merged.shape}")

    cnn_cols = [c for c in merged.columns if c.startswith("cnn_emb_")]
    hybrid_feature_cols = TABULAR_COLS + SATELLITE_COLS + cnn_cols

    years = sorted(merged["Year"].unique())
    models_to_run = {
        "Hybrid - XGBoost": xgb.XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=42),
        "Hybrid - LightGBM": lgb.LGBMRegressor(n_estimators=120, max_depth=3, learning_rate=0.04, num_leaves=8, random_state=42, verbosity=-1),
        "Hybrid - Extra Trees": ExtraTreesRegressor(n_estimators=150, max_depth=6, random_state=42),
        "Hybrid - Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        "Hybrid - Random Forest": RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    }

    hybrid_results = {}
    oof_predictions_dict = {m: [] for m in models_to_run}

    for test_year in years:
        train_df = merged[merged["Year"] != test_year].copy()
        test_df = merged[merged["Year"] == test_year].copy()

        X_train = train_df[hybrid_feature_cols].values
        y_train = train_df["Yield_Tonne_per_Hectare"].values
        X_test = test_df[hybrid_feature_cols].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in models_to_run.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            res = test_df[["District", "Year", "Yield_Tonne_per_Hectare"]].copy()
            res["Predicted"] = preds
            oof_predictions_dict[name].append(res)

    print("\n--- Hybrid Multimodal Model Results (LOYO Cross Validation) ---")
    print(f"{'Model Name':<30} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7} | {'MAPE(%)':<7}")
    print("-" * 65)

    oof_dfs = {}
    for name in models_to_run:
        oof_df = pd.concat(oof_predictions_dict[name], ignore_index=True)
        metrics = compute_metrics(oof_df["Yield_Tonne_per_Hectare"].values, oof_df["Predicted"].values)
        hybrid_results[name] = metrics
        oof_dfs[name] = oof_df
        print(f"{name:<30} | {metrics['MAE']:<7.4f} | {metrics['RMSE']:<7.4f} | {metrics['R2']:<7.4f} | {metrics['MAPE_pct']:<7.2f}")

    # Build Weighted Multimodal Ensemble (Combining PyTorch Deep NN + Hybrid XGBoost + Hybrid LightGBM)
    # PyTorch NN oof predictions are in data/processed/multimodal_nn_results.json or can be computed:
    xgb_preds = oof_dfs["Hybrid - XGBoost"]["Predicted"].values
    lgb_preds = oof_dfs["Hybrid - LightGBM"]["Predicted"].values
    et_preds = oof_dfs["Hybrid - Extra Trees"]["Predicted"].values
    y_true = oof_dfs["Hybrid - XGBoost"]["Yield_Tonne_per_Hectare"].values

    # Ensemble: 40% Hybrid XGBoost + 40% Hybrid LightGBM + 20% Extra Trees
    ensemble_preds = 0.40 * xgb_preds + 0.40 * lgb_preds + 0.20 * et_preds
    ensemble_metrics = compute_metrics(y_true, ensemble_preds)
    hybrid_results["Multimodal Ensemble (Hybrid GBDT)"] = ensemble_metrics
    print(f"{'Multimodal Ensemble (Hybrid GBDT)':<30} | {ensemble_metrics['MAE']:<7.4f} | {ensemble_metrics['RMSE']:<7.4f} | {ensemble_metrics['R2']:<7.4f} | {ensemble_metrics['MAPE_pct']:<7.2f}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(hybrid_results, f, indent=2)

    # Save out-of-fold predictions for visualization
    best_oof = oof_dfs["Hybrid - XGBoost"].copy()
    best_oof["Ensemble_Predicted"] = ensemble_preds
    best_oof.to_csv(HYBRID_PREDS_CSV, index=False)
    print(f"\nHybrid modeling complete! Results saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
