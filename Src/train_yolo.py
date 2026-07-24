“””
Train YOLOv8 on the PCB defect dataset (6 classes: missing_hole, mouse_bite,
open_circuit, short, spur, spurious_copper).

Why YOLOv8 here:

- PCB defects are small, localized regions on a large board image.
- YOLOv8’s anchor-free detection head handles small-object detection well
  compared to older YOLO versions, which matters since mouse_bite / spur
  defects can be just a few pixels.
- Fast enough for a factory inspection line (real-time or near-real-time
  per-board scanning).

Usage:
python train_yolo.py –data ../data/pcb_dataset.yaml –epochs 100 –imgsz 640

NOTE (honest limitation): Not executed in this sandbox — no GPU available
here, and the actual Roboflow PCB dataset has not been downloaded (network
access in this environment is restricted to package registries, not
Roboflow). Run this on Colab/Kaggle with a GPU runtime after downloading
the dataset export in YOLOv8 format. Expect ~1-2 hrs on a T4 GPU for
100 epochs on the ~1400-image PKU-PCB dataset at yolov8n/s scale.
“””

import argparse

from ultralytics import YOLO

def train(data_yaml: str, epochs: int, imgsz: int, model_size: str, batch: int):
# Start from COCO-pretrained weights (transfer learning) rather than
# training from scratch — PCB defect datasets are small (~1000-3000
# images), so pretrained low-level features (edges, textures) from COCO
# give a large head start.
model = YOLO(f”yolov8{model_size}.pt”)

```
results = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    patience=20,           # early stopping if val mAP plateaus
    project="pcb_runs",
    name="pcb_defect_yolov8",
    # PCB images are flat, rigid boards photographed top-down —
    # so we disable augmentations that don't make physical sense here.
    degrees=5.0,            # small rotation only (camera/board misalignment)
    flipud=0.5,              # boards can be imaged either side up
    fliplr=0.5,
    mosaic=1.0,               # helps with small-object detection
    mixup=0.0,                # not useful for rigid structured boards
    perspective=0.0,          # avoid unrealistic warping of a flat board
)

metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

best_path = f"pcb_runs/pcb_defect_yolov8/weights/best.pt"
print(f"Best weights saved to: {best_path}")
print("Copy this to ../models/yolo_pcb_best.pt for use in the Streamlit app.")

return results
```

if **name** == “**main**”:
parser = argparse.ArgumentParser(description=“Train YOLOv8 for PCB defect detection”)
parser.add_argument(”–data”, type=str, default=”../data/pcb_dataset.yaml”)
parser.add_argument(”–epochs”, type=int, default=100)
parser.add_argument(”–imgsz”, type=int, default=640)
parser.add_argument(”–model_size”, type=str, default=“n”, choices=[“n”, “s”, “m”, “l”, “x”],
help=“n=nano (fastest, good for inspection-line speed), s=small, etc.”)
parser.add_argument(”–batch”, type=int, default=16)
args = parser.parse_args()

```
train(args.data, args.epochs, args.imgsz, args.model_size, args.batch)
```
