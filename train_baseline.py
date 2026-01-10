import os
import time
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

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
        y = float(row["label"])
        img = Image.open(find_image_path(image_id)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([y], dtype=torch.float32)  # shape (1,)

def make_loaders(batch_size=32):
    train_csv = os.path.join("ml_data", "train.csv")
    val_csv = os.path.join("ml_data", "val.csv")

    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = HamDataset(train_csv, transform=train_tf)
    val_ds = HamDataset(val_csv, transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_dl, val_dl

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dl, val_dl = make_loaders(batch_size=16)

    # EfficientNet-B0 (szybki baseline)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)  # logit
    model = model.to(device)

    # imbalance: pos_weight = Nneg/Npos
    # train split ma benign=7122, melanoma=890
    pos_weight = torch.tensor([7122/890], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def run_epoch(dl, train: bool):
        model.train(train)
        total_loss = 0.0
        n = 0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
        return total_loss / max(n, 1)

    t0 = time.time()
    train_loss = run_epoch(train_dl, train=True)
    val_loss = run_epoch(val_dl, train=False)
    print(f"Epoch 1/1 | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={(time.time()-t0):.1f}s")

    os.makedirs("artifacts", exist_ok=True)
    out_path = os.path.join("artifacts", "efficientnet_b0_baseline.pt")
    torch.save({"model_state": model.state_dict()}, out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
