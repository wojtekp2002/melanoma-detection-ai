import os
import time
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

#paths
DATA_DIR = "data"
IMG_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

TRAIN_CSV = os.path.join("ml_data", "train.csv")
VAL_CSV   = os.path.join("ml_data", "val.csv")

OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_PATH = os.path.join(OUT_DIR, "efficientnet_b0_best.pt")

#dataset
def find_image_path(image_id: str) -> str:
    fn = image_id + ".jpg"
    p1 = os.path.join(IMG_DIR_1, fn)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(IMG_DIR_2, fn)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Missing image: {image_id}")

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
        return img, torch.tensor([y], dtype=torch.float32)

def make_loaders(batch_size=16):
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

    train_ds = HamDataset(TRAIN_CSV, transform=train_tf)
    val_ds   = HamDataset(VAL_CSV, transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl

#train utils 
def run_epoch(model, dl, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)

    total_loss, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)

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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(device)

    # pos_weight from TRAIN split counts (known from your output)
    pos_weight = torch.tensor([7122/890], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    train_dl, val_dl = make_loaders(batch_size=16)

    best_val = float("inf")
    patience = 3
    bad_epochs = 0
    epochs = 10

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_dl, criterion, optimizer=optimizer, device=device)
        val_loss   = run_epoch(model, val_dl, criterion, optimizer=None, device=device)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.2e} | time={time.time()-t0:.1f}s")

        # save best
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save({"model_state": model.state_dict()}, BEST_PATH)
            print("  ✅ Saved best:", BEST_PATH)
        else:
            bad_epochs += 1
            print(f"  ⏳ No improvement ({bad_epochs}/{patience})")

        if bad_epochs >= patience:
            print("Early stopping.")
            break

    print("Best val_loss:", best_val)

if __name__ == "__main__":
    main()
