import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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
        # safety: jeśli label nie ma, zróbmy go z dx
        if "label" not in self.df.columns:
            self.df["label"] = (self.df["dx"] == "mel").astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label = int(row["label"])
        img_path = find_image_path(image_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    train_csv = os.path.join("ml_data", "train.csv")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = HamDataset(train_csv, transform=transform)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    x, y = next(iter(dl))
    print("Batch X shape:", tuple(x.shape))   
    print("Batch y shape:", tuple(y.shape))   
    print("y values:", y.tolist())
