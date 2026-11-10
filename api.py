"""FastAPI backend. Place at repo root. Run: uvicorn api:app --reload --port 8000"""
import base64
import io
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sic.predict import YieldPredictor, resolve_tile_path, TABULAR_COLS  # noqa: E402

SIC = ROOT / "sic"
CSV = SIC / "data" / "processed" / "multimodal_features_extracted.csv"
MAE = 0.3951
YCOL = "Yield_Tonne_per_Hectare"
SLIDERS = ["HeatStressDays_gt35C", "Rain_Total_mm", "Temp_Mean_C", "Soil_pH", "Clay_pct"]

df = pd.read_csv(CSV)
TCOL = "tile_path" if "tile_path" in df.columns else "tile_file"
predictor = YieldPredictor()

app = FastAPI(title="Crop Yield API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_tile(row) -> np.ndarray:
    arr = np.load(resolve_tile_path(SIC, str(row[TCOL]))).astype("float32")
    return np.moveaxis(arr, 0, -1) if arr.shape[0] == 6 else arr  # -> HWC


def png(img, cmap=None, vmin=None, vmax=None) -> str:
    buf = io.BytesIO()
    plt.imsave(buf, img, cmap=cmap, vmin=vmin, vmax=vmax, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def get_row(district, year):
    m = df[(df["District"] == district) & (df["Year"] == year)]
    if m.empty:
        raise HTTPException(404, "District/year not found")
    return m.iloc[0]


@lru_cache(maxsize=512)
def base_pred(district: str, year: int) -> float:
    row = get_row(district, year)
    return predictor.predict_single({c: float(row[c]) for c in TABULAR_COLS}, load_tile(row))


@app.get("/api/meta")
def meta():
    years = {d: sorted(int(y) for y in g["Year"].unique()) for d, g in df.groupby("District")}
    ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in SLIDERS}
    return {"districts": sorted(years), "years": years, "ranges": ranges, "mae": MAE}


class Req(BaseModel):
    district: str
    year: int
    overrides: dict[str, float] = Field(default_factory=dict)


@app.post("/api/predict")
def predict(req: Req):
    row = get_row(req.district, req.year)
    base = {c: float(row[c]) for c in TABULAR_COLS}
    used = {**base, **{k: v for k, v in req.overrides.items() if k in base}}
    tile = load_tile(row)
    pred = predictor.predict_single(used, tile)

    rgb = np.clip(tile[..., :3] / 10000.0, 0, None)
    rgb = np.clip(rgb / (np.percentile(rgb, 98) + 1e-6), 0, 1)

    d = df[df["District"] == req.district].sort_values("Year")
    history = [{"year": int(r["Year"]), "actual": float(r[YCOL]), "predicted": base_pred(req.district, int(r["Year"]))}
               for _, r in d.iterrows()]
    yr = df[df["Year"] == req.year]
    districts = [{"district": r["District"], "actual": float(r[YCOL])} for _, r in yr.iterrows()]

    return {
        "prediction": pred,
        "actual": float(row[YCOL]),
        "mae": MAE,
        "base": base,
        "history": history,
        "districts": sorted(districts, key=lambda x: x["actual"]),
        "images": {
            "rgb": png(rgb),
            "ndvi": png(tile[..., 4], "RdYlGn", -0.2, 0.9),
            "ndwi": png(tile[..., 5], "Blues"),
        },
    }