from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

DATA_URL = "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part3B_Training_Data.zip"
LABELS_URL = "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"

ROOT = Path("dataset/raw/isic2016")
ZIP_NAME = "ISBI2016_ISIC_Part3B_Training_Data.zip"
LABELS_NAME = "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"


def download(url: str, destination: Path, validate_zip: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        if validate_zip:
            try:
                with zipfile.ZipFile(destination) as archive:
                    bad = archive.testzip()
                    if bad is None:
                        print(f"Already downloaded and verified: {destination}")
                        return
                    print(f"Existing ZIP failed integrity check at: {bad}")
            except (zipfile.BadZipFile, OSError):
                print(f"Existing file is not a valid ZIP: {destination}")
            destination.unlink(missing_ok=True)
        else:
            print(f"Already downloaded: {destination}")
            return

    print(f"Downloading {url}")
    temp = destination.with_suffix(destination.suffix + ".part")
    temp.unlink(missing_ok=True)

    request = Request(url, headers={"User-Agent": "DIF-SEG/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temp.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(f"\rProgress: {percent:6.2f}%", end="", flush=True)
        print()
        if temp.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty.")
        temp.replace(destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    if validate_zip:
        try:
            with zipfile.ZipFile(destination) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"Corrupt ZIP member: {bad}")
        except Exception:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "The downloaded dataset is not a valid ZIP archive. "
                "The partial/corrupt file was removed. Run the downloader again."
            )

    print(f"Saved and verified: {destination}")


def main() -> None:
    download(DATA_URL, ROOT / ZIP_NAME, validate_zip=True)
    download(LABELS_URL, ROOT / LABELS_NAME)
    print("\nISIC 2016 Part 3B files are ready.")
    print("Next: python -m scripts.prepare_isic2016")


if __name__ == "__main__":
    main()
