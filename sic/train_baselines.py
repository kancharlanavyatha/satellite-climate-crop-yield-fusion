"""Train and evaluate baseline ML models for Crop Yield Prediction.

Evaluates models using Leave-One-Year-Out (LOYO) cross-validation to prevent
temporal data leakage.

Models evaluated:
1. Ridge Regression (Tabular Environmental Data)
2. Random Forest Regressor (Tabular Environmental Data)
3. XGBoost Regressor (Tabular Environmental Data)
4. LightGBM Regressor (Tabular Environmental Data)
5. Random Forest (Satellite Indices Only)
6. XGBoost (Satellite Indices Only)
7. Combined Random Forest (Tabular + Satellite Indices)
8. Combined XGBoost (Tabular + Satellite Indices)
9. Combined LightGBM (Tabular + Satellite Indices)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "processed" / "multimodal_features_extracted.csv"
RESULTS_JSON = ROOT / "data" / "processed" / "baseline_results.json"

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

COMBINED_COLS = TABULAR_COLS + SATELLITE_COLS


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


def get_model_instance(model_name: str, random_state: int = 42):
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    elif model_name == "rf":
        return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    elif model_name == "extratrees":
        return ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    elif model_name == "xgb":
        return xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=random_state)
    elif model_name == "lgb":
        return lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=random_state, verbosity=-1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate_loyo(df: pd.DataFrame, feature_cols: list[str], model_type: str) -> tuple[dict, pd.DataFrame]:
    years = sorted(df["Year"].unique())
    oof_predictions = []

    for test_year in years:
        train_df = df[df["Year"] != test_year].copy()
        test_df = df[df["Year"] == test_year].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df["Yield_Tonne_per_Hectare"].values
        X_test = test_df[feature_cols].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = get_model_instance(model_type)
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        
        test_res = test_df[["District", "Year", "Yield_Tonne_per_Hectare"]].copy()
        test_res["Predicted"] = preds
        oof_predictions.append(test_res)

    oof_df = pd.concat(oof_predictions, ignore_index=True)
    metrics = compute_metrics(oof_df["Yield_Tonne_per_Hectare"].values, oof_df["Predicted"].values)
    return metrics, oof_df


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data file missing: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    print(f"Loaded dataset with {len(df)} samples across years {sorted(df['Year'].unique())}.\n")

    experiments = [
        ("Tabular - Ridge", TABULAR_COLS, "ridge"),
        ("Tabular - Random Forest", TABULAR_COLS, "rf"),
        ("Tabular - XGBoost", TABULAR_COLS, "xgb"),
        ("Tabular - LightGBM", TABULAR_COLS, "lgb"),
        ("Satellite - Random Forest", SATELLITE_COLS, "rf"),
        ("Satellite - XGBoost", SATELLITE_COLS, "xgb"),
        ("Combined - Random Forest", COMBINED_COLS, "rf"),
        ("Combined - Extra Trees", COMBINED_COLS, "extratrees"),
        ("Combined - XGBoost", COMBINED_COLS, "xgb"),
        ("Combined - LightGBM", COMBINED_COLS, "lgb"),
    ]

    all_results = {}
    print(f"{'Experiment':<30} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7} | {'MAPE(%)':<7}")
    print("-" * 65)

    for name, cols, model_type in experiments:
        metrics, _ = evaluate_loyo(df, cols, model_type)
        all_results[name] = metrics
        print(f"{name:<30} | {metrics['MAE']:<7.4f} | {metrics['RMSE']:<7.4f} | {metrics['R2']:<7.4f} | {metrics['MAPE_pct']:<7.2f}")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBaseline evaluation complete! Results saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
