"""
Download CBIS-DDSM data exclusively from TCIA.

- CSVs  : direct HTTP download (CC BY 3.0, no auth required)
- Images: NBIA API via tcia_utils (public collection, no auth required)

Usage:
    python src/preprocessing/download_data.py                    # all
    python src/preprocessing/download_data.py --source csv
    python src/preprocessing/download_data.py --source images
    python src/preprocessing/download_data.py --verify-only
"""

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import RAW_CSV_DIR, RAW_IMAGES_DIR


# ── CSV direct download (TCIA, CC BY 3.0) ────────────────────────────────────

TCIA_CSV_BASE = (
    "https://wiki.cancerimagingarchive.net/download/attachments/22516629"
)

CSVS = {
    "mass_case_description_train_set.csv": f"{TCIA_CSV_BASE}/mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv":  f"{TCIA_CSV_BASE}/mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv": f"{TCIA_CSV_BASE}/calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv":  f"{TCIA_CSV_BASE}/calc_case_description_test_set.csv",
}

EXPECTED_CSVS = list(CSVS.keys())


def download_csvs():
    RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in CSVS.items():
        dest = RAW_CSV_DIR / filename

        if dest.exists():
            print(f"  [SKIP] {filename} (already exists)")
            continue

        print(f"  [GET]  {filename}")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"         → {dest}")

    print("[CSV] Done.\n")


# ── Image download via NBIA (tcia_utils) ─────────────────────────────────────

COLLECTION = "CBIS-DDSM"


def download_images():
    try:
        from tcia_utils import nbia
    except ImportError:
        print("ERROR: tcia_utils not installed. Run: pip install tcia_utils")
        sys.exit(1)

    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[NBIA] Fetching series list for collection: {COLLECTION}")
    series_df = nbia.getSeries(collection=COLLECTION)

    if series_df is None or len(series_df) == 0:
        print("ERROR: No series returned. Check network / TCIA status.")
        sys.exit(1)

    print(f"[NBIA] Found {len(series_df)} series → {RAW_IMAGES_DIR}")
    print("       Full dataset ~163 GB. This will take a long time.\n")

    nbia.downloadSeries(
        series_data=series_df,
        path=str(RAW_IMAGES_DIR),
    )
    print("[NBIA] Download complete.\n")


# ── Verify ────────────────────────────────────────────────────────────────────

def verify_downloads():
    print("\n── Verification ─────────────────────────────────────")

    for csv_name in EXPECTED_CSVS:
        path = RAW_CSV_DIR / csv_name
        status = "OK" if path.exists() else "MISSING"
        size = f"({path.stat().st_size // 1024} KB)" if path.exists() else ""
        print(f"  [{status}] {csv_name} {size}")

    dicom_count = len(list(RAW_IMAGES_DIR.rglob("*.dcm")))
    label = "OK" if dicom_count > 0 else "MISSING"
    print(f"\n  [{label}]   DICOM files: {dicom_count}")

    print("─────────────────────────────────────────────────────\n")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CBIS-DDSM from TCIA")
    parser.add_argument(
        "--source",
        choices=["csv", "images", "all"],
        default="all",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_downloads()
        sys.exit(0)

    if args.source in ("csv", "all"):
        download_csvs()

    if args.source in ("images", "all"):
        download_images()

    verify_downloads()
