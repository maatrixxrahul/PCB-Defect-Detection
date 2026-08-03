"""
Streamlit app: PCB Defect Detection (YOLOv8 + Autoencoder Anomaly Detection)

User uploads a PCB image. App runs:
  1. YOLOv8 -> bounding boxes for known defect types (missing_hole, mouse_bite,
     open_circuit, short, spur, spurious_copper)
  2. Autoencoder -> reconstruction-error heatmap for ANY unusual region,
     including defect types YOLO wasn't trained on.
Final verdict combines both: DEFECT if either branch flags something.

Run:
    streamlit run app.py

NOTE: If model weight files are not found (models/ folder empty because
training hasn't been run yet), the app runs in DEMO MODE — it still shows
the UI and pipeline, but with a clear on-screen notice that results are
placeholders until real trained weights are supplied.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from inspector import PCBInspector, InspectionResult

st.set_page_config(page_title="PCB Defect Detection", page_icon="🔍", layout="wide")

MODELS_DIR = Path(__file__).parent.parent / "models"
YOLO_WEIGHTS = MODELS_DIR / "yolo_pcb_best.pt"
AE_WEIGHTS = MODELS_DIR / "autoencoder_best.pt"
THRESHOLD_JSON = MODELS_DIR / "threshold.json"

DEFECT_COLORS = {
    "missing_hole": "#FF3B3B",
    "mouse_bite": "#FF9F1C",
    "open_circuit": "#FFD23F",
    "short": "#E71D36",
    "spur": "#8E44AD",
    "spurious_copper": "#3A86FF",
}


@st.cache_resource
def load_inspector():
    return PCBInspector(
        yolo_weights_path=str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else None,
        autoencoder_weights_path=str(AE_WEIGHTS) if AE_WEIGHTS.exists() else None,
        threshold_json_path=str(THRESHOLD_JSON) if THRESHOLD_JSON.exists() else None,
    )


def draw_boxes(image: Image.Image, detections) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        color = DEFECT_COLORS.get(det.class_name, "#00FF00")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det.class_name} {det.confidence:.2f}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, max(0, y1 - 18)), label, fill="black")
    return img


def heatmap_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay the anomaly heatmap on the original image using a red-hot colormap."""
    base = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    base_arr = np.array(base).astype(float)

    # simple red-hot colormap: heat -> (R high, G mid, B low)
    heat_rgb = np.zeros((*heatmap.shape, 3))
    heat_rgb[..., 0] = np.clip(heatmap * 2.0, 0, 1) * 255          # red ramps fast
    heat_rgb[..., 1] = np.clip(heatmap * 2.0 - 0.5, 0, 1) * 255    # green kicks in at high heat
    heat_rgb[..., 2] = 0

    blended = base_arr * (1 - alpha) + heat_rgb * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def render_verdict_banner(result: InspectionResult, demo_mode: bool):
    if demo_mode:
        st.warning(
            "⚠️ DEMO MODE — model weights not found in `models/`. "
            "Showing pipeline with an untrained/placeholder response. "
            "Train the models (see `src/train_yolo.py` and `src/train_autoencoder.py`) "
            "and drop weights into `models/` to get real results."
        )

    if result.verdict == "DEFECT":
        st.error(f"### 🔴 Verdict: DEFECT DETECTED")
    else:
        st.success(f"### 🟢 Verdict: CLEAN")

    col1, col2, col3 = st.columns(3)
    col1.metric("YOLO Detections", len(result.yolo_detections))
    col2.metric("Anomaly Score", f"{result.anomaly_score_value:.4f}")
    col3.metric("Anomaly Threshold", f"{result.anomaly_threshold:.4f}" if result.anomaly_threshold else "not calibrated")


def main():
    st.title("🔍 PCB Defect Detection")
    st.caption(
        "Upload a PCB image. The app runs **YOLOv8 object detection** (known defect types) "
        "and a **convolutional autoencoder anomaly detector** (catches unusual/unseen regions) "
        "side by side."
    )

    with st.sidebar:
        st.header("Settings")
        yolo_conf = st.slider("YOLO confidence threshold", 0.05, 0.95, 0.25, 0.05)
        st.markdown("---")
        st.subheader("Defect classes (YOLO)")
        for name, color in DEFECT_COLORS.items():
            st.markdown(
                f"<span style='color:{color}'>⬤</span> {name.replace('_', ' ').title()}",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.caption(
            "**How it works:** YOLO flags known defect types with a bounding box. "
            "The autoencoder was trained only on clean PCBs, so it reconstructs "
            "clean regions well and defective regions poorly — the resulting "
            "reconstruction-error heatmap highlights anomalies even if YOLO "
            "wasn't trained on that specific defect type."
        )

    uploaded_file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is None:
        st.info("Upload a PCB board image to run inspection.")
        return

    image = Image.open(uploaded_file)
    inspector = load_inspector()
    demo_mode = inspector.yolo_model is None and inspector.autoencoder is None

    with st.spinner("Running inspection..."):
        result = inspector.inspect(image, yolo_conf=yolo_conf)

    render_verdict_banner(result, demo_mode)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Original", "YOLO Detections", "Anomaly Heatmap"])

    with tab1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with tab2:
        if result.yolo_detections:
            annotated = draw_boxes(image, result.yolo_detections)
            st.image(annotated, caption="YOLO bounding boxes", use_container_width=True)
            st.table([
                {"Defect": d.class_name, "Confidence": f"{d.confidence:.2%}"}
                for d in result.yolo_detections
            ])
        else:
            st.write("No YOLO detections above the confidence threshold." if inspector.yolo_model
                      else "YOLO model not loaded (no trained weights found).")

    with tab3:
        if result.heatmap is not None:
            overlay = heatmap_overlay(image, result.heatmap)
            st.image(overlay, caption="Reconstruction-error heatmap (brighter = more anomalous)",
                      use_container_width=True)
        else:
            st.write("Autoencoder model not loaded (no trained weights found).")

    st.markdown("---")
    st.caption(
        "Verdict logic: flagged as **DEFECT** if YOLO finds any known defect type "
        "*or* the autoencoder's reconstruction error exceeds the calibrated threshold. "
        "This favors catching real defects over avoiding false alarms — appropriate "
        "for a quality-control gate where a missed defect is far more costly than an "
        "extra manual review."
    )


if __name__ == "__main__":
    main()
