“””
Train the ConvAutoencoder on CLEAN (non-defective) PCB images only.

Why only clean images?
The whole point of this anomaly-detection branch is that the model should
never see a defect during training. That way, at inference, ANY pattern
it hasn’t seen (i.e. any defect, even novel/unseen types) produces a large
reconstruction error. If we trained on defective images too, the model
would learn to reconstruct defects “normally” and lose its detection power.

Usage:
python train_autoencoder.py –data_dir data/clean_pcbs –epochs 50

Expected data layout:
data/clean_pcbs/
img_0001.jpg
img_0002.jpg
…

NOTE (honest limitation): This script is written and structured to be
correct and ready to run, but has NOT been executed end-to-end on a real
PCB dataset in this sandbox (no GPU here, and no dataset downloaded).
It should be run on Colab/Kaggle with a GPU runtime and the actual
Roboflow “PCB Defects” clean-image subset (or any clean-PCB image folder).
“””

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from autoencoder import ConvAutoencoder

class CleanPCBDataset(Dataset):
“”“Loads all images in a folder (assumed to be defect-free PCBs).”””

```
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

def __init__(self, data_dir: str, image_size: int = 256):
    self.data_dir = Path(data_dir)
    self.paths = sorted(
        p for p in self.data_dir.glob("*") if p.suffix.lower() in self.IMG_EXTENSIONS
    )
    if len(self.paths) == 0:
        raise FileNotFoundError(
            f"No images found in {data_dir}. Expected clean PCB images "
            f"with extensions {self.IMG_EXTENSIONS}."
        )
    self.transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # scales to [0, 1], matches Sigmoid output range
    ])

def __len__(self):
    return len(self.paths)

def __getitem__(self, idx):
    img = Image.open(self.paths[idx]).convert("RGB")
    return self.transform(img)
```

def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str):
device = torch.device(“cuda” if torch.cuda.is_available() else “cpu”)
print(f”Using device: {device}”)

```
dataset = CleanPCBDataset(data_dir)
val_size = max(1, int(0.1 * len(dataset)))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

model = ConvAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

best_val_loss = float("inf")
os.makedirs(Path(out_path).parent, exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            val_loss += loss.item() * batch.size(0)
    val_loss /= len(val_ds)

    print(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), out_path)
        print(f"  -> saved new best model (val_loss={val_loss:.5f}) to {out_path}")

print("Training complete.")
print(f"Best val_loss: {best_val_loss:.5f}")
print(
    "Next step: run calibrate_threshold.py on a held-out set of clean images "
    "to set the anomaly-score cutoff (mean + 3*std) used at inference time."
)
```

if **name** == “**main**”:
parser = argparse.ArgumentParser(description=“Train PCB anomaly-detection autoencoder”)
parser.add_argument(”–data_dir”, type=str, required=True, help=“Folder of clean PCB images”)
parser.add_argument(”–epochs”, type=int, default=50)
parser.add_argument(”–batch_size”, type=int, default=16)
parser.add_argument(”–lr”, type=float, default=1e-3)
parser.add_argument(”–out_path”, type=str, default=”../models/autoencoder_best.pt”)
args = parser.parse_args()

```
train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out_path)
```
