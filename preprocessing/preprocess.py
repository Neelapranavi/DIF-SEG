from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = (256, 256)


def preprocess_image(image, size=IMAGE_SIZE):
    """Convert an image to normalized grayscale model input."""
    image = image.convert("L").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[None, ...]


def load_image(path, size=IMAGE_SIZE):
    return preprocess_image(Image.open(path), size=size)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
