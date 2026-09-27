# Crop Yield Prediction Model Evaluation Report (Person 2 - Modeling & Prediction)

## Executive Summary
This report summarizes the predictive performance of all machine learning and deep learning models evaluated for district-level Kharif Rice crop yield prediction in Andhra Pradesh (2017–2021). All models were evaluated using strict **Leave-One-Year-Out (LOYO) Cross Validation** to ensure temporal robustness and zero data leakage.

### Champion Model: **PyTorch Deep Multimodal YieldNet (Champion)**
- **$R^2$ Score**: `0.4496`
- **Mean Absolute Error (MAE)**: `0.3951 Tonnes/Hectare`
- **Root Mean Squared Error (RMSE)**: `0.4912 Tonnes/Hectare`
- **Mean Absolute Percentage Error (MAPE)**: `14.12%`

---

## Model Benchmark Comparison Table

| Model Architecture | Modality | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| PyTorch Deep Multimodal YieldNet (Champion) | Deep Multimodal (CNN Satellite + MLP Tabular) | 0.3951 | 0.4912 | 0.4496 | 14.12% |
| Tabular - XGBoost | Unimodal / Tabular Baseline | 0.4510 | 0.5458 | 0.3203 | 16.00% |
| Tabular - LightGBM | Unimodal / Tabular Baseline | 0.4431 | 0.5529 | 0.3027 | 15.57% |
| Tabular - Random Forest | Unimodal / Tabular Baseline | 0.4564 | 0.5592 | 0.2867 | 15.90% |
| Combined - LightGBM | Unimodal / Tabular Baseline | 0.4517 | 0.5677 | 0.2646 | 15.99% |
| Combined - Extra Trees | Unimodal / Tabular Baseline | 0.4939 | 0.5879 | 0.2114 | 16.90% |
| Combined - XGBoost | Unimodal / Tabular Baseline | 0.4698 | 0.5888 | 0.2091 | 16.35% |
| Multimodal Ensemble (Hybrid GBDT) | Hybrid (CNN Embeddings + GBDT) | 0.4895 | 0.5938 | 0.1956 | 16.87% |
| Hybrid - LightGBM | Hybrid (CNN Embeddings + GBDT) | 0.4949 | 0.6022 | 0.1728 | 17.02% |
| Hybrid - XGBoost | Hybrid (CNN Embeddings + GBDT) | 0.4924 | 0.6022 | 0.1728 | 16.99% |
| Combined - Random Forest | Unimodal / Tabular Baseline | 0.4887 | 0.6037 | 0.1685 | 16.84% |
| Tabular - Ridge | Unimodal / Tabular Baseline | 0.5037 | 0.6170 | 0.1315 | 17.75% |
| Hybrid - Gradient Boosting | Hybrid (CNN Embeddings + GBDT) | 0.4910 | 0.6200 | 0.1231 | 17.22% |
| Hybrid - Random Forest | Hybrid (CNN Embeddings + GBDT) | 0.5032 | 0.6208 | 0.1207 | 17.49% |
| Hybrid - Extra Trees | Hybrid (CNN Embeddings + GBDT) | 0.5144 | 0.6221 | 0.1170 | 17.71% |
| Satellite - XGBoost | Unimodal / Tabular Baseline | 0.5788 | 0.7050 | -0.1338 | 19.90% |
| Satellite - Random Forest | Unimodal / Tabular Baseline | 0.5898 | 0.7133 | -0.1608 | 20.27% |

---

## Key Insights & Discussion
1. **Unimodal vs. Multimodal Integration**:
   - Tabular-only environmental models (XGBoost / LightGBM) achieved $R^2 \approx 0.30 - 0.32$.
   - Satellite-only scalar features performed poorly when isolated ($R^2 < 0$), demonstrating that spectral indices require weather and soil context to accurately infer yield.
   - The **PyTorch Deep Multimodal YieldNet** achieved the best overall performance ($R^2 = 0.4068$, $MAE = 0.4087$ t/ha) by jointly encoding 2D Sentinel-2 spatial features via a CNN backbone and tabular environmental factors via an MLP branch.

2. **Modality Attribution**:
   - Weather & Climate Stress (Rainfall seasonality, heat stress days $>35^\circ\text{C}$, mean temperature) contributed **55.03%** of overall feature importance.
   - Satellite Spectral Vegetation Indices (NDVI percentiles, EVI, SAVI, NDWI) contributed **23.19%**.
   - Soil & Terrain factors (Slope, Elevation, Clay content, pH) contributed **20.47%**.
