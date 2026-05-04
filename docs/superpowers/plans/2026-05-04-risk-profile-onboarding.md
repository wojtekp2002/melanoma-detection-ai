# Risk Profile Onboarding — Backend Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `/predict` endpoint to accept clinical risk factors alongside the image and use them to dynamically adjust the detection threshold.

**Architecture:** Extract risk scoring to a pure `compute_clinical_risk(profile)` function (tested with pytest). The `/predict` endpoint accepts an optional `risk_profile` JSON string as a form field, parses it, and adjusts the decision threshold. Falls back to default threshold (0.50) if no profile is provided.

**Tech Stack:** FastAPI, pydantic, pytest

**Related:** Frontend plan in `melanoma-detection-app/docs/superpowers/plans/2026-05-04-risk-profile-onboarding.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `api/main.py` | Modify | RiskProfile model, compute_clinical_risk, extended /predict |
| `api/test_risk_scoring.py` | Create | pytest unit tests for scoring logic |

---

## Task B1: Risk scoring logic (TDD)

**Files:**
- Modify: `api/main.py`
- Create: `api/test_risk_scoring.py`

- [ ] **Step 1: Write failing tests first**

```python
# api/test_risk_scoring.py
import pytest
from main import RiskProfile, compute_clinical_risk


def test_no_risk_factors_returns_low():
    profile = RiskProfile()
    level, threshold = compute_clinical_risk(profile)
    assert level == "low"
    assert threshold == 0.50


def test_family_history_skin_cancer_alone_is_medium():
    profile = RiskProfile(family_history_skin_cancer=True)
    level, threshold = compute_clinical_risk(profile)
    assert level == "medium"
    assert threshold == 0.42


def test_family_history_and_atypical_moles_is_high():
    profile = RiskProfile(family_history_skin_cancer=True, atypical_moles=True)
    level, threshold = compute_clinical_risk(profile)
    assert level == "high"
    assert threshold == 0.35


def test_phototype_I_adds_score():
    profile = RiskProfile(skin_phototype="I")
    level, _ = compute_clinical_risk(profile)
    assert level == "medium"


def test_phototype_III_no_extra_score():
    profile = RiskProfile(skin_phototype="III")
    level, threshold = compute_clinical_risk(profile)
    assert level == "low"
    assert threshold == 0.50


def test_age_over_50_adds_score():
    profile = RiskProfile(age=55, family_history_other_cancer=True)
    level, _ = compute_clinical_risk(profile)
    assert level == "medium"


def test_all_factors_high():
    profile = RiskProfile(
        family_history_skin_cancer=True,
        atypical_moles=True,
        many_moles=True,
        skin_phototype="I",
        age=60,
    )
    level, threshold = compute_clinical_risk(profile)
    assert level == "high"
    assert threshold == 0.35
```

- [ ] **Step 2: Run tests — expect ImportError (nothing defined yet)**

```bash
.venv/bin/pytest api/test_risk_scoring.py -v
```

Expected: `ImportError: cannot import name 'RiskProfile'`

- [ ] **Step 3: Add RiskProfile + compute_clinical_risk to main.py**

Add below the FastAPI imports, before model loading:

```python
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
```

- [ ] **Step 4: Run tests — expect all 7 PASS**

```bash
.venv/bin/pytest api/test_risk_scoring.py -v
```

Expected:
```
test_no_risk_factors_returns_low PASSED
test_family_history_skin_cancer_alone_is_medium PASSED
test_family_history_and_atypical_moles_is_high PASSED
test_phototype_I_adds_score PASSED
test_phototype_III_no_extra_score PASSED
test_age_over_50_adds_score PASSED
test_all_factors_high PASSED
7 passed
```

- [ ] **Step 5: Commit**

```bash
git add api/main.py api/test_risk_scoring.py
git commit -m "feat: add RiskProfile model and compute_clinical_risk with tests"
```

---

## Task B2: Extend /predict endpoint

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Add Form to fastapi import**

Change:
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
```
To:
```python
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
```

- [ ] **Step 2: Replace /predict function**

```python
import json

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
            pass  # malformed profile → fall back to default threshold

    label: Literal["low_risk", "high_risk"] = "high_risk" if prob >= threshold else "low_risk"

    return {
        "probability": prob,
        "threshold": threshold,
        "clinical_risk_level": clinical_risk_level,
        "label": label,
        "disclaimer": "This is not a medical diagnosis. Consult a dermatologist.",
    }
```

- [ ] **Step 3: Confirm tests still pass**

```bash
.venv/bin/pytest api/test_risk_scoring.py -v
```

Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add api/main.py
git commit -m "feat: extend /predict to accept and apply risk_profile"
```
