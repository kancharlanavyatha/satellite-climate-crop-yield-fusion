# Satellite & Climate Crop Yield Fusion


A **Multimodal Deep Learning System** for predicting district-level crop yields by fusing **Sentinel-2 multispectral satellite imagery (6 bands: RGB, NIR, NDVI, NDWI)** with **environmental weather, climate stress, and soil features**.

---



##  Model Benchmark Performance (Leave-One-Year-Out CV)

All models were evaluated using strict **Leave-One-Year-Out (LOYO) Cross-Validation** across 5 folds (2017–2021) to prevent temporal data leakage:

| Model Architecture | Modality | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
|  **PyTorch Deep Multimodal YieldNet (Champion)** | **Deep Multimodal (CNN Satellite + MLP Tabular)** | **0.3951** | **0.4912** | **0.4496** | **14.12%** |
| Tabular - XGBoost | Unimodal / Tabular Baseline | 0.4510 | 0.5458 | 0.3203 | 16.00% |
| Tabular - LightGBM | Unimodal / Tabular Baseline | 0.4431 | 0.5529 | 0.3027 | 15.57% |
| Tabular - Random Forest | Unimodal / Tabular Baseline | 0.4564 | 0.5592 | 0.2867 | 15.90% |
| Tabular - Ridge | Unimodal / Tabular Baseline | 0.5037 | 0.6170 | 0.1315 | 17.75% |

---

##  Modality Attribution & Feature Importance

Feature importance analysis shows that combining environmental constraints with multispectral satellite observations yields the highest accuracy:
- **Weather & Climate Stress (55.03%)**: Heat stress days ($>35^\circ\text{C}$), mean temperature, rainfall seasonality index, and early/mid-season rainfall shares.
- **Satellite Spectral Indices (23.19%)**: NDVI percentiles (p10, p90, median), NDWI, EVI, SAVI, and GCVI.
- **Soil & Terrain (20.47%)**: Slope, elevation, clay content, organic carbon, and soil pH.
- **Agriculture / Sown Area (1.31%)**: Total sown area (hectares).

---

##  Quickstart & Inference Usage

### Installation
```bash
git clone https://github.com/kancharlanavyatha/satellite-climate-crop-yield-fusion.git
cd satellite-climate-crop-yield-fusion
pip install torch numpy pandas scikit-learn xgboost lightgbm matplotlib
```

### Running Model Inference
```python
from sic.predict import YieldPredictor

# Initialize predictor (loads trained PyTorch weights & scaler automatically)
predictor = YieldPredictor()

# Compute yield prediction for a district-year sample
predicted_yield = predictor.predict_single(tabular_input_dict, "sic/data/processed/tiles/Anantapur_2017.npy")
print(f"Predicted Crop Yield: {predicted_yield:.4f} Tonnes/Hectare")
```

---

##  Repository Structure

```
.
├── README.md
├── api.py                              # FastAPI prediction service
├── frontend/                           # React + Vite dashboard
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                     # Dashboard UI
│       ├── main.jsx                    # Frontend entry point
│       └── styles.css
└── sic/
    ├── predict.py                      # Loads model artifacts and runs predictions
    ├── clean_yield_area.py             # Preprocesses crop-yield labels
    ├── clean_senitel_tiles.py          # Prepares Sentinel-2 image tiles
    ├── feature_engineering.py          # Builds weather, climate, and soil features
    ├── image_feature_engineering.py    # Joins image-tile metadata to tabular features
    ├── extract_image_features.py       # Extracts satellite-index statistics
    ├── train_baselines.py              # Trains/evaluates tabular baseline models
    ├── train_multimodal_nn.py          # Defines and trains the multimodal model
    ├── evaluate_and_compare.py         # Compares model performance
    ├── export_pipeline.py              # Exports the trained model artifacts
    ├── earth_engine/                   # Google Earth Engine export scripts
    ├── models/                         # Trained weights, scaler, and model metadata
    └── data/
        ├── raw/                        # Source and intermediate input datasets
        └── processed/
            ├── multimodal_features_extracted.csv  # API input records
            └── tiles/                  # Sentinel-2 .npy tiles used for inference
```

`frontend/node_modules/`, Python caches, and other generated files are local build/runtime artifacts and are not part of the project structure.
