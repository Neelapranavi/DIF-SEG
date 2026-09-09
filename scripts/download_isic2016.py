from pathlib import Path
from urllib.request import urlretrieve


DATA_URL = "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part3B_Training_Data.zip"
LABELS_URL = "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"

ROOT = Path("dataset/raw/isic2016")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already downloaded: {destination}")
        return
    print(f"Downloading {url}")
    urlretrieve(url, destination)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    download(DATA_URL, ROOT / "ISBI2016_ISIC_Part3B_Training_Data.zip")
    download(LABELS_URL, ROOT / "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv")
    print("\nISIC 2016 Part 3B files are ready.")
    print("Next: python -m scripts.prepare_isic2016")
