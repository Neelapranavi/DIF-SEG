from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1] / "dataset" / "demo"
IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"


def main(count=6, size=256):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for index in range(count):
        image = Image.fromarray(rng.normal(80, 18, (size, size)).clip(0, 255).astype(np.uint8))
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        x0, y0 = 55 + index * 3, 65
        x1, y1 = 170 + index * 2, 185
        draw.ellipse((x0, y0, x1, y1), fill=255)
        image_array = np.asarray(image, dtype=np.int16)
        mask_array = np.asarray(mask, dtype=np.int16)
        image_array = np.clip(image_array + (mask_array > 0) * 65, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(IMAGE_DIR / f"demo_{index:03d}.png")
        mask.save(MASK_DIR / f"demo_{index:03d}.png")

    print(f"Generated {count} demo image/mask pairs in {ROOT}")


if __name__ == "__main__":
    main()
