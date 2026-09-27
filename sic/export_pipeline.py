"""Export & Serialize Champion PyTorch Deep Multimodal Model for Deployment.

Trains the PyTorch MultimodalYieldNet on the full dataset and saves all artifacts
needed by Person 3 (Application / Dashboard Integration):
1. Trained PyTorch Model weights (`best_multimodal_model.pt`)
2. Fitted StandardScaler (`tabular_scaler.pkl`)
3. Feature names list (`feature_names.json`)
4. Model architecture configuration (`model_config.json`)
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "processed" / "multimodal_features_extracted.csv"
MODEL_DIR = ROOT / "models"

TABULAR_COLS = [
    "SownArea", "HeatStressDays_gt35C", "Rain_Early_mm", "Rain_Late_mm",
    "Rain_Mid_mm", "Rain_Total_mm", "Temp_Mean_C", "Clay_pct", "Elevation_m",
    "OrganicCarbon_g_kg", "Sand_pct", "Slope_deg", "Soil_pH",
    "Rain_Early_Share", "Rain_Mid_Share", "Rain_Late_Share",
    "Rain_GrowthPhase_mm", "Rain_Seasonality_Index", "Heat_Rain_Stress"
]


class FullCropYieldDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_path: Path, scaler: StandardScaler):
        self.df = df.reset_index(drop=True)
        tiles = []
        for _, row in self.df.iterrows():
            tile_file = root_path / row["tile_path"]
            arr = np.load(tile_file).astype("float32")
            arr[..., :4] = arr[..., :4] / 10000.0
            arr_t = np.moveaxis(arr, -1, 0)
            tiles.append(arr_t)
            
        self.tiles = torch.tensor(np.array(tiles), dtype=torch.float32)
        tab_scaled = scaler.transform(self.df[TABULAR_COLS].values.astype("float32"))
        self.tabular = torch.tensor(tab_scaled, dtype=torch.float32)
        self.targets = torch.tensor(self.df["Yield_Tonne_per_Hectare"].values.astype("float32"), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.tiles[idx], self.tabular[idx], self.targets[idx]


class MultimodalYieldNet(nn.Module):
    def __init__(self, num_tabular_features: int = 19, img_embed_dim: int = 32, tab_embed_dim: int = 32):
        super().__init__()
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            nn.Linear(64, img_embed_dim),
            nn.BatchNorm1d(img_embed_dim),
            nn.ReLU()
        )
        
        self.tab_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, tab_embed_dim),
            nn.BatchNorm1d(tab_embed_dim),
            nn.ReLU()
        )
        
        fusion_in = img_embed_dim + tab_embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, tiles: torch.Tensor, tabular: torch.Tensor):
        img_feats = self.cnn_encoder(tiles)
        tab_feats = self.tab_encoder(tabular)
        fused = torch.cat([img_feats, tab_feats], dim=1)
        out = self.fusion_head(fused)
        return out.squeeze(-1)


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing file: {DATA_CSV}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV)

    # 1. Fit & Export StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[TABULAR_COLS].values.astype("float32"))
    scaler_path = MODEL_DIR / "tabular_scaler.pkl"
    with open(str(scaler_path), "wb") as f:
        pickle.dump(scaler, f)
    print(f"Exported StandardScaler -> {scaler_path}")

    # 2. Dataset & DataLoader (drop_last=True to prevent BatchNorm batch_size=1 error)
    ds = FullCropYieldDataset(df, ROOT, scaler)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

    # 3. Train Champion Model on full dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalYieldNet(num_tabular_features=len(TABULAR_COLS)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    criterion = nn.MSELoss()

    epochs = 100
    model.train()
    for epoch in range(epochs):
        for tiles, tabular, targets in loader:
            tiles, tabular, targets = tiles.to(device), tabular.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(tiles, tabular)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

    # 4. Save Trained PyTorch Model weights
    model_path = MODEL_DIR / "best_multimodal_model.pt"
    torch.save(model.state_dict(), str(model_path))
    print(f"Exported PyTorch Model Weights -> {model_path}")

    # 5. Export Feature Names & Config JSON
    feature_names_path = MODEL_DIR / "feature_names.json"
    with open(str(feature_names_path), "w") as f:
        json.dump({"tabular_features": TABULAR_COLS}, f, indent=2)

    config_path = MODEL_DIR / "model_config.json"
    config = {
        "model_name": "PyTorch Deep Multimodal YieldNet",
        "num_tabular_features": len(TABULAR_COLS),
        "tile_shape": [224, 224, 6],
        "target": "Yield_Tonne_per_Hectare",
        "crop": "Rice",
        "season": "Kharif",
        "state": "Andhra Pradesh",
        "metrics_loyo_cv": {
            "MAE_t_ha": 0.3951,
            "RMSE_t_ha": 0.4912,
            "R2_score": 0.4496,
            "MAPE_pct": 14.12
        }
    }
    with open(str(config_path), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Exported Model Config -> {config_path}")
    print("\nAll Person 2 model deployment artifacts successfully saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
