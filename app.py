from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from segmentation.unet import UNet

st.set_page_config(page_title="DIF-SEG", page_icon="🩺", layout="wide")

WEIGHTS = Path("models/unet_best.pt")
IMAGE_SIZE = (256, 256)


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
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = (probability >= 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask), float(probability.mean()), device.type


st.title("DIF-SEG")
st.caption("Data-Driven Medical Image Detection & Segmentation")
st.info("Research and decision-support prototype only. It is not a clinical diagnostic system.")

uploaded = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded:
    image = Image.open(uploaded).convert("L")
    original, result = st.columns(2)

    with original:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with result:
        st.subheader("Segmentation")
        if not WEIGHTS.exists():
            st.warning("No trained U-Net checkpoint was found.")
            st.code("python -m segmentation.train", language="bash")
            st.caption("Place your paired training images and masks in dataset/ before training.")
        else:
            with st.spinner("Running U-Net inference..."):
                mask, score, device = segment(image)
            st.image(mask, caption="Predicted binary mask", use_container_width=True)
            st.metric("Mean prediction probability", f"{score:.3f}")
            st.caption(f"Inference device: {device}")
            st.download_button(
                "Download mask",
                data=mask.tobytes(),
                file_name="dif_seg_mask.png",
                mime="image/png",
            )

st.divider()
st.subheader("Research pipeline")
st.markdown("**Input → Preprocessing → Detection/Segmentation → Evaluation**")

with st.expander("About DIF-SEG"):
    st.write(
        "DIF-SEG is a research project for data-driven medical image analysis. "
        "The repository includes U-Net segmentation, evaluation utilities, and an "
        "experimental diffusion module for data enhancement."
    )
    st.write("Model performance must be reported from a labeled test set; this app does not invent clinical metrics.")
