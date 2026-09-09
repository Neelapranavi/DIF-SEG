from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from segmentation.unet import UNet
from evaluation.metrics import dice_score, iou_score

IMAGE_SIZE = (256, 256)
WEIGHTS = Path("models/unet_best.pt")

st.set_page_config(page_title="DIF-SEG", page_icon="🩺", layout="wide")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    return model, device


def segment(image):
    model, device = load_model()
    resized = image.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)[None, None].to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu()
    mask = (probability >= 0.5).numpy().astype(np.uint8) * 255
    return Image.fromarray(mask), probability, device.type


st.title("DIF-SEG")
st.caption("Data-Driven Medical Image Detection & Segmentation")
st.info("Research and decision-support prototype only. Not a clinical diagnostic system.")

with st.sidebar:
    st.header("Configuration")
    threshold = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
    st.caption("The trained checkpoint is loaded from models/unet_best.pt.")

uploaded = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded:
    image = Image.open(uploaded).convert("L")
    left, right = st.columns(2)
    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Predicted segmentation")
        if not WEIGHTS.exists():
            st.warning("No trained U-Net checkpoint found.")
            st.code("python -m segmentation.train", language="bash")
            st.caption("Train the model with your paired images and masks, then place the checkpoint at models/unet_best.pt.")
        else:
            with st.spinner("Running U-Net inference..."):
                model, device = load_model()
                resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                array = np.asarray(resized, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array)[None, None].to(device)
                with torch.inference_mode():
                    probability = torch.sigmoid(model(tensor))[0, 0].cpu()
                mask_array = (probability >= threshold).numpy().astype(np.uint8) * 255
                mask = Image.fromarray(mask_array)

            st.image(mask, caption="Binary segmentation mask", use_container_width=True)
            st.metric("Mean prediction probability", f"{probability.mean().item():.3f}")
            st.caption(f"Inference device: {device.type}")
            st.download_button("Download mask", data=mask_array.tobytes(), file_name="dif_seg_mask.png", mime="image/png")

            st.subheader("Evaluation")
            st.caption("Upload a matching ground-truth mask to calculate Dice and IoU.")
            gt_file = st.file_uploader("Ground-truth mask (optional)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="ground_truth")
            if gt_file:
                gt = Image.open(gt_file).convert("L").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
                gt_tensor = torch.from_numpy((np.asarray(gt, dtype=np.float32) / 255.0) > 0.5).float()
                pred_tensor = torch.from_numpy((mask_array > 0).astype(np.float32))
                st.write(f"**Dice:** {dice_score(pred_tensor, gt_tensor):.4f}")
                st.write(f"**IoU:** {iou_score(pred_tensor, gt_tensor):.4f}")
else:
    st.markdown("### Workflow")
    st.markdown("**Upload → Preprocess → U-Net segmentation → Evaluate → Download**")
    st.write("Upload an image to begin.")

st.divider()
st.subheader("About DIF-SEG")
st.write("DIF-SEG is a research project for data-driven medical image detection and segmentation, with experimental diffusion-based data enhancement.")
st.caption("Use labeled test data for performance reporting. Predictions from this prototype should not be used as a medical diagnosis.")
