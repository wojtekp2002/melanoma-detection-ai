import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = r"data"
CSV_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

OUT_DIR = os.path.join("ml_data")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["label"] = (df["dx"] == "mel").astype(int)

# 80% train, 10% val, 10% test (stratyfikacja po label)
train_df, tmp_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    tmp_df, test_size=0.50, random_state=42, stratify=tmp_df["label"]
)

train_path = os.path.join(OUT_DIR, "train.csv")
val_path   = os.path.join(OUT_DIR, "val.csv")
test_path  = os.path.join(OUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

def report(name, d):
    counts = d["label"].value_counts()
    benign = int(counts.get(0, 0))
    mel = int(counts.get(1, 0))
    print(f"{name}: {len(d)} | benign={benign} | melanoma={mel} | mel%={(mel/len(d))*100:.2f}%")

print("Zapisano:")
print(" -", train_path)
print(" -", val_path)
print(" -", test_path)
print()
report("TRAIN", train_df)
report("VAL  ", val_df)
report("TEST ", test_df)
