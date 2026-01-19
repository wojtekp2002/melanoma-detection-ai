import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import roc_auc_score, confusion_matrix

DATA_DIR = "data"
IMG_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

def find_image_path(image_id: str) -> str:
    fn = image_id + ".jpg"
    p1 = os.path.join(IMG_DIR_1, fn)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(IMG_DIR_2, fn)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Nie znaleziono obrazu dla image_id={image_id}")

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
        if self.transform:
            img = self.transform(img)
        return img, y

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

    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            y_prob.extend(prob)
            y_true.extend(y)

    auc = roc_auc_score(y_true, y_prob)

    # próg 0.5 (na start)
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)      # sensitivity
    specificity = tn / (tn + fp + 1e-9)

    report = (
        f"AUC: {auc:.4f}\n"
        f"Threshold: 0.50\n"
        f"TP={tp} FP={fp} TN={tn} FN={fn}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall/Sensitivity: {recall:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
    )

    print("\n" + report)

    os.makedirs("artifacts", exist_ok=True)
    out_txt = os.path.join("artifacts", "val_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved:", out_txt)

if __name__ == "__main__":
    main()
