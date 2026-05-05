import io
import json
import os
from typing import Literal

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class RiskProfile(BaseModel):
    age: int | None = None
    skin_phototype: str | None = None
    family_history_skin_cancer: bool = False
    family_history_other_cancer: bool = False
    had_severe_sunburns: bool = False
    frequent_sun_exposure: bool = False
    uses_tanning_beds: bool = False
    many_moles: bool = False
    atypical_moles: bool = False
    very_fair_skin: bool = False


RISK_WEIGHTS = {
    "family_history_skin_cancer": 3,
    "atypical_moles": 3,
    "family_history_other_cancer": 2,
    "many_moles": 2,
    "had_severe_sunburns": 1,
    "uses_tanning_beds": 1,
    "frequent_sun_exposure": 1,
    "very_fair_skin": 1,
}

PHOTOTYPE_LOW_RISK = {"I", "II"}


def compute_clinical_risk(profile: RiskProfile) -> tuple[str, float]:
    """Returns (clinical_risk_level, threshold)."""
    score = 0

    for field, weight in RISK_WEIGHTS.items():
        if getattr(profile, field, False):
            score += weight

    if profile.skin_phototype in PHOTOTYPE_LOW_RISK:
        score += 2

    if profile.age is not None and profile.age > 50:
        score += 1

    if score <= 2:
        return "low", 0.50
    elif score <= 5:
        return "medium", 0.42
    else:
        return "high", 0.35

# --- konfiguracja ---
MODEL_PATH = os.path.join("artifacts", "efficientnet_b0_best.pt")
THRESHOLD = 0.50  # wybrany próg

app = FastAPI(title="Melanoma Detection API", version="0.1")

#pozwolenie RN/Expo łączyć się lokalnie
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def build_model() -> torch.nn.Module:
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, 1)
    return m

#ładowanie modelu
model = build_model().to(device)
model.eval()

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Nie znaleziono pliku modelu: {MODEL_PATH}")

state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)["model_state"]
model.load_state_dict(state)

@app.get("/health")
def health():
    return {"status": "ok", "device": device, "threshold": THRESHOLD}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    risk_profile: str | None = Form(default=None),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Wgraj obraz JPG/PNG/WEBP.")

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Plik za duży (max 10 MB).")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Niepoprawny plik obrazu.")

    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        prob = torch.sigmoid(logit).item()

    clinical_risk_level: str | None = None
    threshold = THRESHOLD

    if risk_profile:
        try:
            profile_data = json.loads(risk_profile)
            profile = RiskProfile(**profile_data)
            clinical_risk_level, threshold = compute_clinical_risk(profile)
        except Exception:
            pass

    label: Literal["low_risk", "high_risk"] = "high_risk" if prob >= threshold else "low_risk"

    return {
        "probability": prob,
        "threshold": threshold,
        "clinical_risk_level": clinical_risk_level,
        "label": label,
        "disclaimer": "This is not a medical diagnosis. Consult a dermatologist.",
    }
