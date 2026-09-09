from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from detection.model import SimpleCNN
from evaluation.metrics import dice_score, iou_score
from segmentation.unet import UNet

IMAGE_SIZE = (256, 256)
SEG_WEIGHTS = Path("models/unet_best.pt")
DET_WEIGHTS = Path("models/detection_best.pt")

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


def preprocess(image, size=IMAGE_SIZE):
    resized = image.convert("L").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, None]


def run_segmentation(image, threshold):
    model, device = load_segmentation_model()
    tensor = preprocess(image).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu()
    mask = (probability >= threshold).numpy().astype(np.uint8) * 255
    return mask, probability, device.type


def run_detection(image):
    model, device = load_detection_model()
    tensor = preprocess(image, (224, 224)).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    label = "Abnormal" if probabilities[1] >= probabilities[0] else "Normal"
    return label, float(probabilities[1]), device.type


st.title("DIF-SEG")
st.caption("Data-Driven Medical Image Detection & Segmentation")
st.info("Research and decision-support prototype only. Not a clinical diagnostic system.")

with st.sidebar:
    st.header("Configuration")
    threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
    st.caption("Model checkpoints are loaded from the models/ directory.")

uploaded = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded:
    image = Image.open(uploaded).convert("L")
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
                st.image(mask, caption="Predicted mask", use_container_width=True)
            st.metric("Mean prediction probability", f"{probability.mean().item():.3f}")
            st.caption(f"Inference device: {device}")
            st.download_button("Download mask", data=mask_array.tobytes(), file_name="dif_seg_mask.png", mime="image/png")

    with evaluation_tab:
        if not SEG_WEIGHTS.exists():
            st.info("Train the segmentation model first.")
        else:
            st.caption("Upload a matching ground-truth mask to calculate segmentation metrics.")
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
    st.markdown("**Upload → Detection → Segmentation → Evaluation → Download**")
    st.write("Upload a medical image to begin.")

st.divider()
st.subheader("About DIF-SEG")
st.write("DIF-SEG is a research project for data-driven medical image analysis, combining binary abnormality detection, U-Net segmentation, evaluation, and an experimental diffusion-based data-enhancement component.")
st.caption("Use labeled test data for performance reporting. Predictions from this prototype must not be treated as a medical diagnosis.")
