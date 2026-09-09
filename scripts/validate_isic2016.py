from collections import Counter
from pathlib import Path
import csv


ROOT = Path("dataset")
IMAGES = ROOT / "images"
MASKS = ROOT / "masks"
DETECTION = ROOT / "detection"
LABELS = ROOT / "raw/isic2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    images = {p.stem for p in IMAGES.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS} if IMAGES.exists() else set()
    masks = {p.stem for p in MASKS.iterdir() if p.is_file() and p.suffix.lower() == ".png"} if MASKS.exists() else set()

    labels = {}
    if LABELS.exists():
        with LABELS.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.reader(handle):
                if len(row) >= 2 and row[0].strip().startswith("ISIC_"):
                    labels[row[0].strip()] = row[1].strip().lower()

    print(f"Segmentation images : {len(images)}")
    print(f"Segmentation masks  : {len(masks)}")
    print(f"Ground-truth labels : {len(labels)}")
    print(f"Image/mask matches   : {len(images & masks)}")
    print(f"Image/label matches  : {len(images & labels.keys())}")
    print(f"Label distribution   : {dict(Counter(labels.values()))}")

    problems = []
    problems.extend(sorted(images - masks)[:10])
    problems.extend(sorted(images - labels.keys())[:10])
    problems.extend(sorted(masks - images)[:10])

    if problems:
        print("Validation: FAILED")
        for item in problems:
            print(" -", item)
        raise SystemExit(1)

    if len(images) == 0:
        raise SystemExit("No prepared ISIC images found. Run the download and preparation commands first.")

    print("Validation: PASSED")


if __name__ == "__main__":
    main()
