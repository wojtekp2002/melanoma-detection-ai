import os
import pandas as pd

DATA_DIR = r"data"
CSV_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

df = pd.read_csv(CSV_PATH)

print("Liczba rekordów:", len(df))
print("\nRozkład dx (wszystkie klasy):")
print(df["dx"].value_counts())

# binary label: mel=1, reszta=0
df["label"] = (df["dx"] == "mel").astype(int)
print("\nBinary label: mel vs reszta")
print(df["label"].value_counts().rename({0: "benign", 1: "melanoma"}))

# sprawdź czy zdjęcia istnieją w part_1 / part_2
img1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
img2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

def exists_image(image_id: str) -> bool:
    fn = image_id + ".jpg"
    return os.path.exists(os.path.join(img1, fn)) or os.path.exists(os.path.join(img2, fn))

missing = [iid for iid in df["image_id"].astype(str).tolist() if not exists_image(iid)]
print("\nBrakujących obrazów:", len(missing))
if missing[:5]:
    print("Przykłady brakujących:", missing[:5])
