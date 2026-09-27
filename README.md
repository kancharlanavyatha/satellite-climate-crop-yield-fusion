# Satellite & Climate Crop Yield Fusion 🌾🛰️

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-008000?style=flat)](https://xgboost.readthedocs.io/)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-Copernicus-003366?style=flat)](https://sentinels.copernicus.eu/)

A **Multimodal Deep Learning & Machine Learning System** for predicting district-level crop yields by fusing **Sentinel-2 multispectral satellite imagery (6 bands: RGB, NIR, NDVI, NDWI)** with **environmental weather, climate stress, and soil characteristics**.

---

## 🏗️ System Workflow & Responsibilities

```
Satellite + Weather/Soil/Other Data
       │
       ▼
Person 1: Data Collection & Feature Engineering (Sentinel Tiles, Soil, Climate Indices)
       │
       ▼
Person 2: ML/DL Modeling (PyTorch Deep Multimodal YieldNet, LOYO CV, Benchmarks)
       │
       ▼
Person 3: Application / Dashboard & Visualization (YieldPredictor API Integration)
       │
       ▼
Final Crop Yield Prediction Result (Tonnes/Hectare)
```

- **👩 Person 1 — Data & Features**: Preprocessed 6-band Sentinel-2 `.npy` tiles, extracted soil parameters (clay, sand, pH, organic carbon), climate stress metrics (heat stress days, rainfall seasonality), and cleaned historical target labels (Andhra Pradesh Kharif Rice, 2017–2021).
- **🧠 Person 2 — Model**: Built unimodal baselines (Ridge, RF, XGBoost, LightGBM), designed the **PyTorch Deep Multimodal Late Fusion Neural Network (`MultimodalYieldNet`)**, performed Leave-One-Year-Out (LOYO) cross-validation, feature importance attribution, and exported deployment artifacts ([models/](sic/models)).
- **💻 Person 3 — Application & Integration**: Dashboard interface connecting inputs to the [predict.py](sic/predict.py) inference engine for interactive prediction and visualization.

---

## 📊 Model Benchmark Performance (Leave-One-Year-Out CV)

All models were evaluated using strict **Leave-One-Year-Out (LOYO) Cross-Validation** across 5 folds (2017–2021) to prevent temporal data leakage and evaluate real-world generalization:

| Model Architecture | Modality | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 🏆 **PyTorch Deep Multimodal YieldNet (Champion)** | **Deep Multimodal (CNN Satellite + MLP Tabular)** | **0.3951** | **0.4912** | **0.4496** | **14.12%** |
| Tabular - XGBoost | Unimodal / Tabular Baseline | 0.4510 | 0.5458 | 0.3203 | 16.00% |
| Tabular - LightGBM | Unimodal / Tabular Baseline | 0.4431 | 0.5529 | 0.3027 | 15.57% |
| Tabular - Random Forest | Unimodal / Tabular Baseline | 0.4564 | 0.5592 | 0.2867 | 15.90% |
| Combined - LightGBM | Tabular + Satellite Stats | 0.4517 | 0.5677 | 0.2646 | 15.99% |
| Combined - Extra Trees | Tabular + Satellite Stats | 0.4939 | 0.5879 | 0.2114 | 16.90% |
| Combined - XGBoost | Tabular + Satellite Stats | 0.4698 | 0.5888 | 0.2091 | 16.35% |
| Multimodal Ensemble (Hybrid GBDT) | Hybrid (CNN Embeddings + GBDT) | 0.4895 | 0.5938 | 0.1956 | 16.87% |
| Hybrid - XGBoost / LightGBM | Hybrid (CNN Embeddings + GBDT) | 0.4924 | 0.6022 | 0.1728 | 16.99% |
| Satellite - XGBoost | Satellite Stats Only | 0.5788 | 0.7050 | -0.1338 | 19.90% |

---

## 🔍 Modality Attribution & Feature Importance

Feature importance analysis shows that combining environmental constraints with multispectral satellite observations yields the highest accuracy:
- **Weather & Climate Stress (55.03%)**: Heat stress days ($>35^\circ\text{C}$), mean temperature, rainfall seasonality index, and early/mid-season rainfall shares.
- **Satellite Spectral Indices (23.19%)**: NDVI percentiles (p10, p90, median), NDWI, EVI, SAVI, and GCVI.
- **Soil & Terrain (20.47%)**: Slope, elevation, clay content, organic carbon, and soil pH.
- **Agriculture / Sown Area (1.31%)**: Total sown area (hectares).

---

## 🚀 Quickstart & Inference Usage

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

## 📁 Repository Structure

```
.
├── README.md                           # Repository documentation
├── sic/
│   ├── clean_yield_area.py             # Preprocesses raw crop yield labels
│   ├── clean_senitel_tiles.py          # Validates & resizes Sentinel-2 GeoTIFF exports to 224x224x6 .npy tiles
│   ├── feature_engineering.py          # Merges weather, soil, and climate stress tabular features
│   ├── image_feature_engineering.py    # Merges tile manifest into tabular feature tables
│   ├── inspect_tiles.py                # Tile inspection & preview generator
│   ├── extract_image_features.py       # Extracts NDVI, NDWI, EVI, SAVI, GCVI statistics
│   ├── train_baselines.py              # LOYO CV for unimodal baseline ML models
│   ├── train_multimodal_nn.py          # PyTorch Deep Multimodal YieldNet implementation
│   ├── train_hybrid_multimodal.py      # Hybrid GBDT modeling with CNN embeddings
│   ├── explain_model.py                # Feature importance & modality attribution script
│   ├── evaluate_and_compare.py         # Benchmarks all model architectures
│   ├── export_pipeline.py              # Trains champion model & exports artifacts to models/
│   ├── predict.py                      # Production inference API for Person 3 application integration
│   ├── earth_engine/                   # Earth Engine JS export scripts
│   ├── data/
│   │   ├── raw/                        # Raw boundary, soil, and weather data
│   │   └── processed/                  # Cleaned features, manifests, benchmark results, & .npy tiles
│   └── models/                         # Exported trained PyTorch model weights & scalers
```
