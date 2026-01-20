import io
import os
from typing import Literal

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

#ładujemy model raz przy starcie
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
async def predict(file: UploadFile = File(...)):
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

    label: Literal["low_risk", "high_risk"] = "high_risk" if prob >= THRESHOLD else "low_risk"

    return {
        "probability": prob,
        "threshold": THRESHOLD,
        "label": label,
        "disclaimer": "This is not a medical diagnosis. Consult a dermatologist.",
    }
