# Project 16: PCB Defect Detection

Dual-pipeline PCB inspection: **YOLOv8 object detection** for known defect
types + a **convolutional autoencoder anomaly detector** for unusual/unseen
regions. Streamlit app lets a user upload a PCB image and get a combined
verdict.

## Why two models?

|          |YOLOv8 Detection                          |Autoencoder Anomaly Detection                 |
|----------|------------------------------------------|----------------------------------------------|
|Trained on|Labeled defect images (6 classes)         |ONLY clean/normal PCB images                  |
|Catches   |Known defect types, precisely localized   |Anything unusual, including novel defect types|
|Output    |Bounding box + class + confidence         |Reconstruction-error heatmap + scalar score   |
|Weakness  |Blind to defect types not in training data|No class label — just “this looks off”        |

Combining both gives broader coverage than either alone, which is standard
practice in industrial visual inspection.

## Defect classes (YOLO)

`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`
— the standard 6-class taxonomy from the widely-used PKU PCB Defect Dataset
(available via Roboflow Universe in YOLOv8 export format).

## Project structure

```
pcb-defect-detection/
├── data/
│   └── pcb_dataset.yaml        # YOLOv8 dataset config (6 defect classes)
├── models/                     # trained weights go here (not included)
├── src/
│   ├── autoencoder.py          # ConvAutoencoder architecture
│   ├── train_autoencoder.py    # trains AE on clean PCB images only
│   ├── calibrate_threshold.py  # sets anomaly-score cutoff from clean val set
│   ├── train_yolo.py           # fine-tunes YOLOv8 on labeled defects
│   ├── inspector.py            # combined inference: loads both models, merges verdict
│   └── app.py                  # Streamlit upload UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training (requires GPU — Colab/Kaggle recommended)

1. **Download dataset**: Get the PCB Defect Dataset from Roboflow Universe,
   exported in YOLOv8 format, into `data/pcb-defect-dataset/`.
1. **Train YOLOv8**:
   
   ```bash
   cd src
   python train_yolo.py --data ../data/pcb_dataset.yaml --epochs 100 --model_size n
   ```
   
   Copy `pcb_runs/pcb_defect_yolov8/weights/best.pt` → `models/yolo_pcb_best.pt`
1. **Train the autoencoder** on a folder of clean (non-defective) PCB images:
   
   ```bash
   python train_autoencoder.py --data_dir ../data/clean_pcbs --epochs 50 --out_path ../models/autoencoder_best.pt
   ```
1. **Calibrate the anomaly threshold** on a held-out clean set:
   
   ```bash
   python calibrate_threshold.py --data_dir ../data/clean_val --model_path ../models/autoencoder_best.pt --out_json ../models/threshold.json
   ```

## Run the app

```bash
cd src
streamlit run app.py
```

If `models/` doesn’t contain trained weights yet, the app still launches in
**demo mode** — full UI and pipeline visible, with an on-screen notice that
results are placeholders until real weights are supplied.

## Honest limitations (what’s verified vs. what needs GPU training)

**Verified in this sandbox (CPU only, no GPU):**

- Autoencoder architecture: forward pass, shapes, reconstruction-error and
  anomaly-score functions — all correct (`torch.Size` checks passed).
- Autoencoder training loop: ran 3 real epochs on synthetic images, loss
  decreased monotonically (0.088 → 0.045), checkpointing confirmed working.
- Threshold calibration script: ran end-to-end, produced a sane
  mean/std/threshold from a synthetic clean-image set.
- YOLOv8 training call: ran a full smoke test with the exact same
  hyperparameters as `train_yolo.py` (augmentation settings, optimizer,
  patience, etc.) on a tiny synthetic labeled dataset — training completed,
  weights saved, validation ran without errors.
- Combined `PCBInspector` class: loaded both a (test) YOLO checkpoint and
  autoencoder checkpoint, ran `.inspect()` end-to-end, produced a correct
  CLEAN verdict on a clean test image with matching score/threshold logic.
- Streamlit app: launched headless, returned HTTP 200, no crash in demo
  mode (no real weights present) — confirms the UI code path and
  cached-resource loading logic are sound.

**NOT done in this sandbox (needs real GPU + real dataset):**

- No real PCB dataset was downloaded (Roboflow is not on this sandbox’s
  network allowlist — only package registries like PyPI/npm/GitHub are
  reachable).
- No real YOLOv8 training run on actual PCB defect images — only a
  synthetic-data smoke test of the training call itself.
- No real autoencoder training on actual clean PCB photographs — only a
  synthetic-image smoke test.
- Actual mAP / precision / recall numbers on the real dataset are unknown
  until trained on real data with a GPU (Colab/Kaggle, ~1-2 hrs for YOLO
  at 100 epochs on ~1400 images with a T4 GPU).

## Deployment notes

- YOLOv8n (nano) recommended for inspection-line speed; step up to `s`/`m`
  if accuracy on small defects (mouse_bite, spur) isn’t sufficient.
- Autoencoder threshold (`k=3` std-dev default) should be re-tuned against
  real production images — false-positive rate depends heavily on lighting/
  camera consistency across your actual imaging setup.
