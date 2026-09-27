"""Inference API for Person 3 (Application / Dashboard Integration).

Provides simple, robust inference functions to load trained model artifacts
and compute crop yield predictions on new inputs.
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models"

TABULAR_COLS = [
    "SownArea", "HeatStressDays_gt35C", "Rain_Early_mm", "Rain_Late_mm",
    "Rain_Mid_mm", "Rain_Total_mm", "Temp_Mean_C", "Clay_pct", "Elevation_m",
    "OrganicCarbon_g_kg", "Sand_pct", "Slope_deg", "Soil_pH",
    "Rain_Early_Share", "Rain_Mid_Share", "Rain_Late_Share",
    "Rain_GrowthPhase_mm", "Rain_Seasonality_Index", "Heat_Rain_Stress"
]


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


class YieldPredictor:
    """Production Inference Engine for Person 3's Dashboard."""

    def __init__(self, model_dir: Path | str = None):
        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR
        else:
            model_dir = Path(model_dir)

        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load scaler
        scaler_path = self.model_dir / "tabular_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        with open(str(scaler_path), "rb") as f:
            self.scaler = pickle.load(f)

        # Load feature names
        names_path = self.model_dir / "feature_names.json"
        if names_path.exists():
            with open(str(names_path), "r") as f:
                self.feature_names = json.load(f).get("tabular_features", TABULAR_COLS)
        else:
            self.feature_names = TABULAR_COLS

        # Load model config
        config_path = self.model_dir / "model_config.json"
        if config_path.exists():
            with open(str(config_path), "r") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Instantiate & load PyTorch Model weights
        model_path = self.model_dir / "best_multimodal_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights: {model_path}")

        self.model = MultimodalYieldNet(num_tabular_features=len(self.feature_names)).to(self.device)
        self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.model.eval()

    def _prepare_tile(self, tile_input: Path | str | np.ndarray) -> torch.Tensor:
        if isinstance(tile_input, (str, Path)):
            tile_path = Path(tile_input)
            if not tile_path.exists():
                raise FileNotFoundError(f"Tile file not found: {tile_path}")
            arr = np.load(tile_path).astype("float32")
        elif isinstance(tile_input, np.ndarray):
            arr = tile_input.astype("float32")
        else:
            raise TypeError("tile_input must be a file path or numpy array.")

        if arr.ndim == 3 and arr.shape == (224, 224, 6):
            arr[..., :4] = arr[..., :4] / 10000.0
            arr = np.moveaxis(arr, -1, 0)
        elif arr.ndim == 3 and arr.shape == (6, 224, 224):
            arr[:4] = arr[:4] / 10000.0
        else:
            raise ValueError(f"Invalid tile shape: {arr.shape}, expected (224, 224, 6) or (6, 224, 224).")

        # Add batch dimension: (1, 6, 224, 224)
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _prepare_tabular(self, tabular_input: dict | pd.Series | pd.DataFrame) -> torch.Tensor:
        if isinstance(tabular_input, dict):
            row_vals = [tabular_input[col] for col in self.feature_names]
            vec = np.array([row_vals], dtype="float32")
        elif isinstance(tabular_input, pd.Series):
            row_vals = [tabular_input[col] for col in self.feature_names]
            vec = np.array([row_vals], dtype="float32")
        elif isinstance(tabular_input, pd.DataFrame):
            vec = tabular_input[self.feature_names].values.astype("float32")
        else:
            raise TypeError("tabular_input must be dict, pd.Series, or pd.DataFrame.")

        scaled = self.scaler.transform(vec)
        return torch.tensor(scaled, dtype=torch.float32).to(self.device)

    def predict_single(self, tabular_input: dict | pd.Series, tile_input: Path | str | np.ndarray) -> float:
        """Compute yield prediction for a single district-year sample.

        Returns:
            predicted_yield_t_ha (float): Yield in Tonnes/Hectare.
        """
        tile_t = self._prepare_tile(tile_input)
        tab_t = self._prepare_tabular(tabular_input)

        with torch.no_grad():
            out = self.model(tile_t, tab_t)
            pred = float(out.cpu().numpy()[0])
        return round(pred, 4)

    def predict_dataframe(self, df: pd.DataFrame, root_dir: Path | str = None) -> np.ndarray:
        """Compute yield predictions for a pandas DataFrame."""
        if root_dir is None:
            root_dir = ROOT
        else:
            root_dir = Path(root_dir)

        preds = []
        for _, row in df.iterrows():
            tile_path = root_dir / row["tile_path"] if "tile_path" in row else row["tile_file"]
            pred = self.predict_single(row, tile_path)
            preds.append(pred)
        return np.array(preds)


def main():
    print("Testing Person 2 Predictor API...")
    predictor = YieldPredictor()

    data_csv = ROOT / "data" / "processed" / "multimodal_features_extracted.csv"
    if data_csv.exists():
        df = pd.read_csv(data_csv)
        sample = df.iloc[0]
        tile_file = ROOT / sample["tile_path"]
        
        pred_yield = predictor.predict_single(sample.to_dict(), tile_file)
        actual_yield = sample["Yield_Tonne_per_Hectare"]

        print(f"\n--- Inference Test ---")
        print(f"District:           {sample['District']} ({int(sample['Year'])})")
        print(f"Predicted Yield:    {pred_yield:.4f} t/ha")
        print(f"Actual Yield:       {actual_yield:.4f} t/ha")
        print(f"Absolute Difference: {abs(pred_yield - actual_yield):.4f} t/ha")
        print("\nPredictor API works cleanly! Person 3 can import YieldPredictor directly from sic/predict.py!")


if __name__ == "__main__":
    main()
