"""
Convert CBIS-DDSM DICOM images → PNG.

Steps per image:
  1. Load DICOM with pydicom
  2. Apply DICOM windowing / VOI LUT if present
  3. Normalize pixel values to [0, 255]
  4. Resize to IMAGE_SIZE
  5. Save as PNG, mirroring source folder structure

Usage:
    python src/preprocessing/dicom_to_png.py
    python src/preprocessing/dicom_to_png.py --workers 8
    python src/preprocessing/dicom_to_png.py --limit 100   # test run
"""

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import IMAGE_SIZE, PROC_IMAGES_DIR, RAW_IMAGES_DIR # type: ignore
# ── Core conversion ──────────────────────────────────────────────────────────
def dicom_to_array(dcm: pydicom.Dataset) -> np.ndarray:
    """
    Read pixel array from DICOM, apply VOI LUT / windowing, return float32 [0,1].
    """
    # Apply VOI LUT (handles WINDOW_CENTER / WINDOW_WIDTH tags if present)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            arr = apply_voi_lut(dcm.pixel_array, dcm)
        except Exception:
            arr = dcm.pixel_array.astype(np.float32)

    arr = arr.astype(np.float32)

    # Photometric inversion (some CBIS-DDSM DICOMs use MONOCHROME1)
    if hasattr(dcm, "PhotometricInterpretation"):
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr

    # Normalize to [0, 1]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)

    return arr


def convert_single(dcm_path: Path, out_path: Path) -> tuple[str, bool, str]:
    """
    Convert one DICOM file to PNG. Returns (path_str, success, error_msg).
    """
    try:
        dcm = pydicom.dcmread(str(dcm_path))
        arr = dicom_to_array(dcm)

        # Scale to uint8
        img_uint8 = (arr * 255).astype(np.uint8)

        # Resize
        img_resized = cv2.resize(
            img_uint8, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img_resized)

        return str(dcm_path), True, ""

    except Exception as e:
        return str(dcm_path), False, str(e)


# ── Batch conversion ─────────────────────────────────────────────────────────

def build_job_list(limit: int | None) -> list[tuple[Path, Path]]:
    """
    Walk RAW_IMAGES_DIR, find all .dcm files, map to output PNG paths.
    Skips already-converted files.
    """
    dcm_files = sorted(RAW_IMAGES_DIR.rglob("*.dcm"))

    if limit:
        dcm_files = dcm_files[:limit]

    jobs = []
    for dcm_path in dcm_files:
        # Mirror folder structure under PROC_IMAGES_DIR
        rel = dcm_path.relative_to(RAW_IMAGES_DIR)
        out_path = PROC_IMAGES_DIR / rel.with_suffix(".png")

        if out_path.exists():
            continue  # already converted

        jobs.append((dcm_path, out_path))

    return jobs


def run_conversion(workers: int = 4, limit: int | None = None):
    jobs = build_job_list(limit)

    if not jobs:
        print("No new DICOM files to convert. All already processed (or raw dir empty).")
        return

    print(f"Converting {len(jobs)} DICOM files → PNG  (workers={workers})")
    print(f"Output dir: {PROC_IMAGES_DIR}\n")

    failed = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert_single, dcm, out): dcm for dcm, out in jobs}

        with tqdm(total=len(jobs), unit="file") as pbar:
            for future in as_completed(futures):
                path_str, success, err = future.result()
                pbar.update(1)
                if not success:
                    failed.append((path_str, err))
                    pbar.set_postfix({"failed": len(failed)})

    print(f"\nDone. Converted: {len(jobs) - len(failed)}  Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for path_str, err in failed:
            print(f"  {path_str}  →  {err}")

        # Write failure log
        log_path = PROC_IMAGES_DIR / "conversion_failures.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            for path_str, err in failed:
                f.write(f"{path_str}\t{err}\n")
        print(f"\nFailure log saved to: {log_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CBIS-DDSM DICOM → PNG")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files (for testing)",
    )
    args = parser.parse_args()

    run_conversion(workers=args.workers, limit=args.limit)
