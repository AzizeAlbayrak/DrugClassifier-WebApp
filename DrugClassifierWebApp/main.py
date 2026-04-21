from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from torch import nn


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "drug_classification_model.pth"
SCALER_PATH = APP_DIR / "scaler.joblib"


class DrugClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)


Sex = Literal[0, 1]
BP = Literal[0, 1, 2]
Cholesterol = Literal[0, 1]


class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: Sex
    bp: BP
    cholesterol: Cholesterol
    na_to_k: float = Field(..., ge=0)


DRUG_LABELS = ["DrugY", "drugA", "drugB", "drugC", "drugX"]

SEX_MAP = {0: 0.0, 1: 1.0}  # 0->Kadın, 1->Erkek
BP_MAP = {0: 0.0, 1: 1.0, 2: 2.0}  # High->0, Low->1, Normal->2
CHOL_MAP = {0: 0.0, 1: 1.0}  # High->0, Normal->1


def _load_model() -> DrugClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH}. "
            f"`drug_classification_model.pth` dosyasını proje dizinine koyun."
        )

    model = DrugClassifier()
    try:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler dosyası bulunamadı: {SCALER_PATH}. "
            f"`scaler.joblib` dosyasını proje dizinine koyun."
        )
    return joblib.load(SCALER_PATH)


model = _load_model()
scaler = None
_scaler_load_error: str | None = None
try:
    scaler = _load_scaler()
except ModuleNotFoundError as e:
    _scaler_load_error = (
        "Scaler yüklenemedi (eksik paket). "
        "Çözüm: `pip install scikit-learn joblib` (gerekirse aynı venv içinde). "
        f"Ayrıntı: {e}"
    )
except Exception as e:
    _scaler_load_error = f"Scaler yüklenemedi. Ayrıntı: {type(e).__name__}: {e}"
app = FastAPI(title="Drug200 Classifier", version="1.0.0")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/predict")
def predict(payload: PredictRequest) -> JSONResponse:
    if scaler is None:
        raise HTTPException(status_code=500, detail=_scaler_load_error or "Scaler yüklenemedi.")

    # Feature order MUST be: [Age, Sex, BP, Cholesterol, Na_to_K]
    raw = np.array(
        [
            [
                float(payload.age),
                float(SEX_MAP[int(payload.sex)]),
                float(BP_MAP[int(payload.bp)]),
                float(CHOL_MAP[int(payload.cholesterol)]),
                float(payload.na_to_k),
            ]
        ],
        dtype=np.float32,
    )
    scaled = scaler.transform(raw).astype(np.float32, copy=False)
    x = torch.from_numpy(scaled)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())

    return JSONResponse(
        {
            "prediction": {"index": idx, "label": DRUG_LABELS[idx]},
            "probabilities": [
                {"label": DRUG_LABELS[i], "prob": float(probs[i].item())}
                for i in range(len(DRUG_LABELS))
            ],
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

