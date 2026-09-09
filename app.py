from io import BytesIO
from pathlib import Path
import json

import numpy as np
import streamlit as st
import torch
from PIL import Image

from detection.model import SimpleCNN
from evaluation.metrics import dice_score, iou_score
from segmentation.unet import UNet

IMAGE_SIZE = (256, 256)
DET_IMAGE_SIZE = (224, 224)
SEG_WEIGHTS = Path("models/unet_best.pt")
DET_WEIGHTS = Path("models/detection_best.pt")
SEG_RESULTS = Path("evaluation_results/segmentation_metrics.json")
DET_RESULTS = Path("evaluation_results/detection_metrics.json")
DIFF_RESULTS = Path("evaluation_results/segmentation_diffusion_metrics.json")
COMPARISON_RESULTS = Path("evaluation_results/segmentation_comparison.json")

st.set_page_config(page_title="DIF-SEG", page_icon="🩺", layout="wide")


@st.cache_resource
def load_segmentation_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(SEG_WEIGHTS, map_location=device))
    model.eval()
    return model, device


@st.cache_resource
def load_detection_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(DET_WEIGHTS, map_location=device))
    model.eval()
    return model, device


def preprocess(image, size=IMAGE_SIZE, grayscale=False):
    mode = "L" if grayscale else "RGB"
    resized = image.convert(mode).resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if grayscale:
        return torch.from_numpy(array)[None, None]
    return torch.from_numpy(array).permute(2, 0, 1)[None]


def run_segmentation(image, threshold):
    model, device = load_segmentation_model()
    tensor = preprocess(image, grayscale=False).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu()
    mask_array = (probability >= threshold).numpy().astype(np.uint8) * 255
    return mask_array, probability, device.type


def run_detection(image):
    model, device = load_detection_model()
    tensor = preprocess(image, DET_IMAGE_SIZE, grayscale=False).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    label = "Abnormal" if probabilities[1] >= probabilities[0] else "Normal"
    return label, float(probabilities[1]), device.type


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_metrics(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


st.title("DIF-SEG")
st.caption("Data-Driven Medical Image Detection & Segmentation")
st.info("Research and decision-support prototype only. Not a clinical diagnostic system.")

with st.sidebar:
    st.header("Configuration")
    threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
    st.caption("Model checkpoints are loaded from the models/ directory.")

uploaded = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Input image")
    st.image(image, width=420)

    detection_tab, segmentation_tab, evaluation_tab = st.tabs(["Detection", "Segmentation", "Evaluation"])

    with detection_tab:
        if not DET_WEIGHTS.exists():
            st.warning("Detection model checkpoint not found.")
            st.code("python -m detection.train", language="bash")
        else:
            with st.spinner("Running abnormality detection..."):
                label, abnormal_probability, device = run_detection(image)
            st.metric("Prediction", label)
            st.metric("Abnormal probability", f"{abnormal_probability:.3f}")
            st.caption(f"Inference device: {device}")

    with segmentation_tab:
        if not SEG_WEIGHTS.exists():
            st.warning("U-Net checkpoint not found.")
            st.code("python -m segmentation.train", language="bash")
        else:
            with st.spinner("Running U-Net segmentation..."):
                mask_array, probability, device = run_segmentation(image, threshold)
            mask = Image.fromarray(mask_array)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Input", use_container_width=True)
            with col2:
                st.image(mask, caption="Predicted lesion mask", use_container_width=True)
            st.metric("Mean prediction probability", f"{probability.mean().item():.3f}")
            st.caption(f"Inference device: {device}")
            st.download_button("Download mask", data=png_bytes(mask), file_name="dif_seg_mask.png", mime="image/png")

    with evaluation_tab:
        st.subheader("Held-out evaluation")
        det_metrics = load_metrics(DET_RESULTS)
        seg_metrics = load_metrics(SEG_RESULTS)
        diff_metrics = load_metrics(DIFF_RESULTS)
        comparison = load_metrics(COMPARISON_RESULTS)

        if det_metrics:
            st.markdown("**Detection results**")
            cols = st.columns(5)
            cols[0].metric("Accuracy", f"{det_metrics['accuracy']:.4f}")
            cols[1].metric("Precision", f"{det_metrics['precision']:.4f}")
            cols[2].metric("Recall", f"{det_metrics['recall_sensitivity']:.4f}")
            cols[3].metric("Specificity", f"{det_metrics['specificity']:.4f}")
            cols[4].metric("F1", f"{det_metrics['f1']:.4f}")
        else:
            st.info("Detection metrics are not available yet. Run: python -m evaluation.evaluate_detection")

        if seg_metrics:
            st.markdown("**Baseline U-Net results**")
            cols = st.columns(4)
            cols[0].metric("Mean Dice", f"{seg_metrics['mean_dice']:.4f}")
            cols[1].metric("Mean IoU", f"{seg_metrics['mean_iou']:.4f}")
            cols[2].metric("Dice std", f"{seg_metrics['std_dice']:.4f}")
            cols[3].metric("IoU std", f"{seg_metrics['std_iou']:.4f}")
        else:
            st.info("Segmentation metrics are not available yet. Run: python -m evaluation.evaluate_segmentation")

        if diff_metrics:
            st.markdown("**Diffusion-inspired augmentation results**")
            cols = st.columns(4)
            cols[0].metric("Mean Dice", f"{diff_metrics['mean_dice']:.4f}")
            cols[1].metric("Mean IoU", f"{diff_metrics['mean_iou']:.4f}")
            cols[2].metric("Dice std", f"{diff_metrics['std_dice']:.4f}")
            cols[3].metric("IoU std", f"{diff_metrics['std_iou']:.4f}")

        if comparison:
            st.markdown("**Baseline vs diffusion-inspired experiment**")
            cols = st.columns(2)
            cols[0].metric("Dice change", f"{comparison['delta_mean_dice']:+.4f}")
            cols[1].metric("IoU change", f"{comparison['delta_mean_iou']:+.4f}")
            st.caption("Positive values indicate improvement over the baseline on the same held-out split.")
        else:
            st.info("Run the diffusion experiment and comparison after training to populate this section.")

        st.divider()
        st.caption("For an individual uploaded image, provide its ground-truth mask to calculate image-level Dice and IoU.")
        if SEG_WEIGHTS.exists():
            gt_file = st.file_uploader("Ground-truth mask (optional)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="evaluation_mask")
            if gt_file:
                mask_array, _, _ = run_segmentation(image, threshold)
                gt = Image.open(gt_file).convert("L").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
                gt_array = (np.asarray(gt, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
                pred_array = (mask_array > 0).astype(np.float32)
                c1, c2 = st.columns(2)
                c1.metric("Dice", f"{dice_score(pred_array, gt_array):.4f}")
                c2.metric("IoU", f"{iou_score(pred_array, gt_array):.4f}")
else:
    st.markdown("### Workflow")
    st.markdown("**Upload → Detection → Segmentation → Evaluation → Compare**")
    st.write("Upload a medical image to begin.")

st.divider()
st.subheader("About DIF-SEG")
st.write("DIF-SEG is a research project for data-driven medical image analysis, combining binary abnormality detection, U-Net segmentation, evaluation, and an experimental diffusion-inspired data-enhancement component.")
st.caption("Use labeled test data for performance reporting. Predictions from this prototype must not be treated as a medical diagnosis.")
