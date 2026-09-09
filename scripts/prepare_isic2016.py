from pathlib import Path
import csv
import shutil
import zipfile

RAW = Path("dataset/raw/isic2016")
ZIP_PATH = RAW / "ISBI2016_ISIC_Part3B_Training_Data.zip"
LABELS_PATH = RAW / "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
IMAGES_DIR = Path("dataset/images")
MASKS_DIR = Path("dataset/masks")
NORMAL_DIR = Path("dataset/detection/normal")
ABNORMAL_DIR = Path("dataset/detection/abnormal")
EXTRACTED = RAW / "extracted"


def prepare() -> None:
    if not ZIP_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Run python -m scripts.download_isic2016 first.")

    try:
        with zipfile.ZipFile(ZIP_PATH) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"Corrupt ZIP member: {bad}")
    except (zipfile.BadZipFile, OSError) as exc:
        raise RuntimeError(
            "The ISIC dataset ZIP is invalid or corrupted. "
            "Delete dataset/raw/isic2016/ISBI2016_ISIC_Part3B_Training_Data.zip "
            "and run python -m scripts.download_isic2016 again."
        ) from exc

    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(EXTRACTED)

    images = {p.stem: p for p in EXTRACTED.rglob("*.jpg")}
    masks = {
        p.stem.replace("_Segmentation", ""): p
        for p in EXTRACTED.rglob("*_Segmentation.png")
    }

    labels = {}
    with LABELS_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2 and row[0].strip().startswith("ISIC_"):
                labels[row[0].strip()] = row[1].strip().lower()

    for directory in (IMAGES_DIR, MASKS_DIR, NORMAL_DIR, ABNORMAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    prepared = 0
    for image_id, image_path in sorted(images.items()):
        mask_path = masks.get(image_id)
        label = labels.get(image_id)
        if mask_path is None or label not in {"benign", "malignant"}:
            continue

        shutil.copy2(image_path, IMAGES_DIR / f"{image_id}.jpg")
        shutil.copy2(mask_path, MASKS_DIR / f"{image_id}.png")
        target = NORMAL_DIR if label == "benign" else ABNORMAL_DIR
        shutil.copy2(image_path, target / f"{image_id}.jpg")
        prepared += 1

    if prepared == 0:
        raise RuntimeError(
            "The ZIP was readable, but no matching image/mask/label cases were found. "
            "Check the downloaded ISIC archive contents."
        )

    print(f"Prepared {prepared} real ISIC cases.")
    print(f"Segmentation images: {IMAGES_DIR}")
    print(f"Segmentation masks:  {MASKS_DIR}")
    print(f"Detection folders:   {NORMAL_DIR.parent}")


if __name__ == "__main__":
    prepare()
