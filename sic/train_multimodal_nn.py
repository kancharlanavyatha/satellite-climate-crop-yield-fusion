"""PyTorch Deep Multimodal Late Fusion Network for Crop Yield Prediction.

Combines 6-band Sentinel-2 satellite tiles (CNN) and tabular environmental data (MLP).
Uses Leave-One-Year-Out (LOYO) cross validation to train and evaluate.
"""

import copy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "processed" / "multimodal_features_extracted.csv"
OUTPUT_NN_JSON = ROOT / "data" / "processed" / "multimodal_nn_results.json"
EMBEDDINGS_CSV = ROOT / "data" / "processed" / "cnn_tabular_embeddings.csv"

TABULAR_COLS = [
    "SownArea", "HeatStressDays_gt35C", "Rain_Early_mm", "Rain_Late_mm",
    "Rain_Mid_mm", "Rain_Total_mm", "Temp_Mean_C", "Clay_pct", "Elevation_m",
    "OrganicCarbon_g_kg", "Sand_pct", "Slope_deg", "Soil_pH",
    "Rain_Early_Share", "Rain_Mid_Share", "Rain_Late_Share",
    "Rain_GrowthPhase_mm", "Rain_Seasonality_Index", "Heat_Rain_Stress"
]


def resolve_tile_path(root: Path, tile_path_str: str) -> Path:
    p1 = root / tile_path_str
    if p1.exists():
        return p1
    p2 = root.parent / tile_path_str
    if p2.exists():
        return p2
    p3 = root / "data" / "processed" / "tiles" / Path(tile_path_str).name
    if p3.exists():
        return p3
    return p1


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


class CropYieldDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_path: Path, tabular_scaler: StandardScaler = None, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.root_path = root_path
        
        # Load satellite tiles
        tiles = []
        for _, row in self.df.iterrows():
            tile_file = resolve_tile_path(self.root_path, row["tile_path"])
            arr = np.load(tile_file).astype("float32") # (224, 224, 6)
            arr[..., :4] = arr[..., :4] / 10000.0
            arr_t = np.moveaxis(arr, -1, 0)
            tiles.append(arr_t)
            
        self.tiles = torch.tensor(np.array(tiles), dtype=torch.float32)
        
        # Process tabular features
        tab_data = self.df[TABULAR_COLS].values.astype("float32")
        if is_train:
            self.scaler = StandardScaler()
            tab_scaled = self.scaler.fit_transform(tab_data)
        else:
            self.scaler = tabular_scaler
            tab_scaled = self.scaler.transform(tab_data)
            
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

    def extract_features(self, tiles: torch.Tensor, tabular: torch.Tensor):
        img_feats = self.cnn_encoder(tiles)
        tab_feats = self.tab_encoder(tabular)
        fused = torch.cat([img_feats, tab_feats], dim=1)
        return img_feats, tab_feats, fused

    def forward(self, tiles: torch.Tensor, tabular: torch.Tensor):
        _, _, fused = self.extract_features(tiles, tabular)
        out = self.fusion_head(fused)
        return out.squeeze(-1)


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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for tiles, tabular, targets in loader:
        tiles, tabular, targets = tiles.to(device), tabular.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(tiles, tabular)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for tiles, tabular, targets in loader:
            tiles, tabular = tiles.to(device), tabular.to(device)
            preds = model(tiles, tabular)
            all_preds.extend(preds.cpu().numpy().tolist())
    return np.array(all_preds)


def main():
    set_seed(42)
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing file: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    years = sorted(df["Year"].unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using PyTorch device: {device}")

    oof_predictions = []
    embedding_records = []

    for test_year in years:
        train_df = df[df["Year"] != test_year].copy()
        test_df = df[df["Year"] == test_year].copy()

        train_ds = CropYieldDataset(train_df, ROOT, is_train=True)
        test_ds = CropYieldDataset(test_df, ROOT, tabular_scaler=train_ds.scaler, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

        model = MultimodalYieldNet(num_tabular_features=len(TABULAR_COLS)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.MSELoss()

        epochs = 100
        best_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if loss < best_loss:
                best_loss = loss
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        preds = eval_epoch(model, test_loader, device)

        res = test_df[["District", "Year", "Yield_Tonne_per_Hectare"]].copy()
        res["Predicted"] = preds
        oof_predictions.append(res)

        # Extract latent CNN embeddings
        model.eval()
        with torch.no_grad():
            for i, (tiles, tabular, _) in enumerate(test_loader):
                tiles, tabular = tiles.to(device), tabular.to(device)
                img_f, _, _ = model.extract_features(tiles, tabular)
                img_f = img_f.cpu().numpy()
                sub_df = test_df.iloc[i * 8 : (i + 1) * 8]
                for k, (_, row) in enumerate(sub_df.iterrows()):
                    rec = {"District": row["District"], "Year": int(row["Year"])}
                    for dim_idx in range(img_f.shape[1]):
                        rec[f"cnn_emb_{dim_idx}"] = float(img_f[k, dim_idx])
                    embedding_records.append(rec)

    oof_df = pd.concat(oof_predictions, ignore_index=True)
    nn_metrics = compute_metrics(oof_df["Yield_Tonne_per_Hectare"].values, oof_df["Predicted"].values)

    print("\n--- PyTorch Deep Multimodal YieldNet (LOYO CV Results) ---")
    print(f"MAE:     {nn_metrics['MAE']:.4f} t/ha")
    print(f"RMSE:    {nn_metrics['RMSE']:.4f} t/ha")
    print(f"R²:      {nn_metrics['R2']:.4f}")
    print(f"MAPE:    {nn_metrics['MAPE_pct']:.2f} %")

    OUTPUT_NN_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_NN_JSON), "w") as f:
        json.dump(nn_metrics, f, indent=2)

    emb_df = pd.DataFrame(embedding_records)
    EMBEDDINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(str(EMBEDDINGS_CSV), index=False)
    print(f"Successfully saved {len(emb_df)} embedding rows to: {EMBEDDINGS_CSV}")


if __name__ == "__main__":
    main()
