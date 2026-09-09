from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = (256, 256)


def preprocess_image(image):
    image = image.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[None, ...]


def load_image(path):
    return preprocess_image(Image.open(path))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
