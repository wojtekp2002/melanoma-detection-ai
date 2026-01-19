import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import confusion_matrix, roc_auc_score

DATA_DIR = "data"
IMG_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

def find_image_path(image_id: str) -> str:
    fn = image_id + ".jpg"
    p1 = os.path.join(IMG_DIR_1, fn)
    if os.path.exists(p1): return p1
    p2 = os.path.join(IMG_DIR_2, fn)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(image_id)

class HamDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        if "label" not in self.df.columns:
            self.df["label"] = (self.df["dx"] == "mel").astype(int)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        y = int(row["label"])
        img = Image.open(find_image_path(image_id)).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

def metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (np.array(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return tp, fp, tn, fn, precision, recall, specificity

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    val_csv = os.path.join("ml_data", "val.csv")
    ckpt_path = os.path.join("artifacts", "efficientnet_b0_best.pt")

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds = HamDataset(val_csv, transform=tf)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    state = torch.load(ckpt_path, map_location="cpu")["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            y_prob.extend(prob)
            y_true.extend(y)

    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC: {auc:.4f}")

    #metryki dla 0.50 kontrola
    tp, fp, tn, fn, prec, rec, spec = metrics_at_threshold(y_true, y_prob, 0.50)
    print("\n@ threshold=0.50")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn} | Precision={prec:.4f} Recall={rec:.4f} Spec={spec:.4f}")

    # ? prog dla recall >= 0.95
    target_recall = 0.95
    best = None

    for thr in np.linspace(0.01, 0.99, 199):  # krok ~0.005
        tp, fp, tn, fn, prec, rec, spec = metrics_at_threshold(y_true, y_prob, float(thr))
        if rec >= target_recall:
            #najwyższa specificity
            cand = (spec, thr, tp, fp, tn, fn, prec, rec)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        print(f"\nNie znaleziono progu dla recall >= {target_recall:.2f}")
    else:
        spec, thr, tp, fp, tn, fn, prec, rec = best
        print(f"\nTarget recall >= {target_recall:.2f}")
        print(f"Recommended threshold: {thr:.3f}")
        print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"Precision={prec:.4f} Recall={rec:.4f} Specificity={spec:.4f}")

if __name__ == "__main__":
    main()
