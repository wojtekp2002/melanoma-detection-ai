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

CKPT_PATH = os.path.join("artifacts", "efficientnet_b0_best.pt")
VAL_CSV = os.path.join("ml_data", "val.csv")
TEST_CSV = os.path.join("ml_data", "test.csv")

def find_image_path(image_id: str) -> str:
    fn = image_id + ".jpg"
    p1 = os.path.join(IMG_DIR_1, fn)
    if os.path.exists(p1): return p1
    p2 = os.path.join(IMG_DIR_2, fn)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(image_id)

class HamDataset(torch.utils.data.Dataset):
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

def metrics(y_true, y_prob, thr: float):
    y_pred = (np.array(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return {"thr": thr, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": precision, "recall": recall, "specificity": specificity}

def infer_probs(csv_path: str, device: str):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds = HamDataset(csv_path, transform=tf)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    state = torch.load(CKPT_PATH, map_location="cpu")["model_state"]
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
    return y_true, y_prob, auc

def find_threshold_for_target_recall(y_true, y_prob, target_recall: float):
    best = None
    for thr in np.linspace(0.01, 0.99, 199):
        m = metrics(y_true, y_prob, float(thr))
        if m["recall"] >= target_recall:
            # wybieramy najwyższą specificity (najmniej FP) przy wymaganym recall
            cand = (m["specificity"], m)
            if best is None or cand[0] > best[0]:
                best = cand
    return None if best is None else best[1]

def print_block(title, auc, rows):
    print("\n" + title)
    print(f"AUC: {auc:.4f}")
    print("thr    TP  FP  TN  FN   precision  recall  specificity")
    for r in rows:
        print(f"{r['thr']:.3f}  {r['TP']:3d} {r['FP']:3d} {r['TN']:3d} {r['FN']:3d}   "
              f"{r['precision']:.4f}    {r['recall']:.4f}   {r['specificity']:.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # VAL
    val_true, val_prob, val_auc = infer_probs(VAL_CSV, device)
    thr_050_val = metrics(val_true, val_prob, 0.50)
    thr_0361_val = metrics(val_true, val_prob, 0.361)
    thr_090_val = find_threshold_for_target_recall(val_true, val_prob, 0.90)  # kompromis

    val_rows = [thr_050_val, thr_0361_val]
    if thr_090_val:
        val_rows.append(thr_090_val)

    # TEST
    test_true, test_prob, test_auc = infer_probs(TEST_CSV, device)
    thr_050_test = metrics(test_true, test_prob, 0.50)
    thr_0361_test = metrics(test_true, test_prob, 0.361)

    test_rows = [thr_050_test, thr_0361_test]
    if thr_090_val:
        test_rows.append(metrics(test_true, test_prob, thr_090_val["thr"]))

    print_block("VAL REPORT", val_auc, val_rows)
    print_block("TEST REPORT", test_auc, test_rows)

    # zapis do pliku
    os.makedirs("artifacts", exist_ok=True)
    out_path = os.path.join("artifacts", "threshold_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        # proste “przekierowanie” tekstu:
        f.write("VAL REPORT\n")
        f.write(f"AUC: {val_auc:.4f}\n")
        for r in val_rows:
            f.write(str(r) + "\n")
        f.write("\nTEST REPORT\n")
        f.write(f"AUC: {test_auc:.4f}\n")
        for r in test_rows:
            f.write(str(r) + "\n")
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
