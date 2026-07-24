“””
Combined inference: run both YOLOv8 detection and autoencoder anomaly
detection on a single uploaded PCB image, and merge results.

Design rationale for the combined verdict:

- YOLO gives precise, labeled, high-confidence detections for KNOWN defect
  types it was trained on.
- The autoencoder gives a heatmap + scalar anomaly score that can catch
  anything unusual, including defect types YOLO was never trained on.
- We report both independently rather than forcing a single fused score,
  because in a real inspection line a human operator wants to see:
  “YOLO found 2 confirmed defects” AND/OR “Autoencoder flagged an unusual
  region YOLO didn’t label — worth a manual look.”
- Final verdict logic: DEFECT if either branch fires. This is deliberately
  conservative (favors recall over precision) which is the right tradeoff
  for quality control — a false alarm costs a few seconds of operator
  review; a missed defect costs a shipped faulty board.
  “””

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from autoencoder import ConvAutoencoder, reconstruction_error_map, anomaly_score

@dataclass
class YoloDetection:
class_name: str
confidence: float
box_xyxy: list  # [x1, y1, x2, y2] in original image pixel coords

@dataclass
class InspectionResult:
yolo_detections: list = field(default_factory=list)
anomaly_score_value: float = 0.0
anomaly_threshold: float = 0.0
is_anomalous: bool = False
heatmap: Optional[np.ndarray] = None  # (H, W) float array, 0-1
verdict: str = “CLEAN”  # “CLEAN” or “DEFECT”

class PCBInspector:
“””
Loads a YOLOv8 detector and a ConvAutoencoder, and runs both on an
input image to produce a combined InspectionResult.

```
If a model file is missing (e.g. not yet trained), that branch is
skipped gracefully rather than crashing — useful for demo mode before
real training data/weights are available.
"""

def __init__(
    self,
    yolo_weights_path: Optional[str] = None,
    autoencoder_weights_path: Optional[str] = None,
    threshold_json_path: Optional[str] = None,
    image_size: int = 256,
):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.image_size = image_size
    self.transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # --- YOLO branch ---
    self.yolo_model = None
    if yolo_weights_path and Path(yolo_weights_path).exists():
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_weights_path)

    # --- Autoencoder branch ---
    self.autoencoder = None
    self.threshold = None
    if autoencoder_weights_path and Path(autoencoder_weights_path).exists():
        self.autoencoder = ConvAutoencoder().to(self.device)
        self.autoencoder.load_state_dict(
            torch.load(autoencoder_weights_path, map_location=self.device)
        )
        self.autoencoder.eval()

    if threshold_json_path and Path(threshold_json_path).exists():
        with open(threshold_json_path) as f:
            self.threshold = json.load(f)["threshold"]

def run_yolo(self, image: Image.Image, conf: float = 0.25) -> list:
    if self.yolo_model is None:
        return []
    results = self.yolo_model.predict(image, conf=conf, verbose=False)
    detections = []
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls.item())
            detections.append(
                YoloDetection(
                    class_name=names[cls_id],
                    confidence=float(box.conf.item()),
                    box_xyxy=box.xyxy[0].tolist(),
                )
            )
    return detections

def run_autoencoder(self, image: Image.Image):
    if self.autoencoder is None:
        return 0.0, None
    img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
    with torch.no_grad():
        recon = self.autoencoder(img_tensor)
        score = anomaly_score(img_tensor, recon).item()
        heatmap_tensor = reconstruction_error_map(img_tensor, recon)
    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    # normalize heatmap to 0-1 for display
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return score, heatmap

def inspect(self, image: Image.Image, yolo_conf: float = 0.25) -> InspectionResult:
    result = InspectionResult()

    result.yolo_detections = self.run_yolo(image, conf=yolo_conf)

    score, heatmap = self.run_autoencoder(image)
    result.anomaly_score_value = score
    result.heatmap = heatmap
    if self.threshold is not None:
        result.anomaly_threshold = self.threshold
        result.is_anomalous = score > self.threshold

    has_yolo_defect = len(result.yolo_detections) > 0
    result.verdict = "DEFECT" if (has_yolo_defect or result.is_anomalous) else "CLEAN"

    return result
```
