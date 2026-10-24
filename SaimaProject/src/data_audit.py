"""
Dataset audit utility for mapped CBIS-DDSM CSV files.

Checks:
1) CSV availability
2) Image path existence for each row
3) Label/class balance summary
4) Basic recommendations for preprocessing and balancing

Usage:
    python src/data_audit.py
    python src/data_audit.py --train-csv data/processed/csv/mass_train_mapped.csv --test-csv data/processed/csv/mass_test_mapped.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def resolve_image_path(project_root: Path, raw_path: str) -> Path:
    p = Path(str(raw_path).strip())
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def audit_split(project_root: Path, csv_path: Path, image_col: str, label_col: str) -> dict:
    result: dict = {
        "csv": str(csv_path),
        "csv_exists": csv_path.exists(),
        "rows": 0,
        "columns": [],
        "missing_required_columns": [],
        "image_exists_count": 0,
        "image_missing_count": 0,
        "image_missing_examples": [],
        "class_counts": {},
        "class_percent": {},
    }

    if not csv_path.exists():
        return result

    df = pd.read_csv(csv_path)
    result["rows"] = int(len(df))
    result["columns"] = list(df.columns)

    for col in (image_col, label_col):
        if col not in df.columns:
            result["missing_required_columns"].append(col)

    if result["missing_required_columns"]:
        return result

    missing_examples = []
    exists_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        resolved = resolve_image_path(project_root, row[image_col])
        if resolved.exists():
            exists_count += 1
        else:
            missing_count += 1
            if len(missing_examples) < 10:
                missing_examples.append(str(resolved))

    result["image_exists_count"] = int(exists_count)
    result["image_missing_count"] = int(missing_count)
    result["image_missing_examples"] = missing_examples

    counts = df[label_col].value_counts(dropna=False)
    perc = (counts / len(df) * 100).round(2)
    result["class_counts"] = {str(k): int(v) for k, v in counts.items()}
    result["class_percent"] = {str(k): float(v) for k, v in perc.items()}

    return result


def make_recommendations(train_result: dict) -> list[str]:
    recs: list[str] = []

    if not train_result.get("csv_exists", False):
        recs.append("Training CSV is missing. Add mapped CSV files first.")
        return recs

    if train_result.get("missing_required_columns"):
        recs.append("CSV schema mismatch. Ensure columns image_path and pathology exist.")
        return recs

    miss = train_result.get("image_missing_count", 0)
    rows = max(train_result.get("rows", 0), 1)
    miss_pct = miss / rows * 100
    if miss > 0:
        recs.append(
            f"Fix image path mapping before training ({miss}/{rows} missing, {miss_pct:.2f}%)."
        )
    else:
        recs.append("Image mapping looks good (0 missing image paths).")

    class_counts = train_result.get("class_counts", {})
    if class_counts:
        vals = [v for v in class_counts.values() if v > 0]
        if vals:
            imbalance_ratio = max(vals) / min(vals)
            if imbalance_ratio >= 2.0:
                recs.append(
                    "Dataset is imbalanced (max/min >= 2). Use class weights and consider oversampling minority class."
                )
            else:
                recs.append("Class distribution is reasonably balanced.")

    recs.append("Do focused EDA: per-class counts, missing-value audit, and path-level duplicate checks.")
    return recs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit mapped CSVs against image files.")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/processed/csv/mass_train_mapped.csv",
        help="Path to training mapped CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/processed/csv/mass_test_mapped.csv",
        help="Path to test mapped CSV",
    )
    parser.add_argument("--image-col", type=str, default="image_path")
    parser.add_argument("--label-col", type=str, default="pathology")
    parser.add_argument(
        "--out-json",
        type=str,
        default="results/data_audit_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_csv = (project_root / args.train_csv).resolve()
    test_csv = (project_root / args.test_csv).resolve()

    train_result = audit_split(project_root, train_csv, args.image_col, args.label_col)
    test_result = audit_split(project_root, test_csv, args.image_col, args.label_col)
    recommendations = make_recommendations(train_result)

    report = {
        "train": train_result,
        "test": test_result,
        "recommendations": recommendations,
    }

    out_path = (project_root / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("DATASET AUDIT SUMMARY")
    print("=" * 72)
    print(f"Train CSV exists: {train_result['csv_exists']} | {train_result['csv']}")
    print(f"Test  CSV exists: {test_result['csv_exists']} | {test_result['csv']}")

    if train_result["csv_exists"] and not train_result["missing_required_columns"]:
        print(
            f"Train images present: {train_result['image_exists_count']} | "
            f"missing: {train_result['image_missing_count']}"
        )
        print(f"Train class counts: {train_result['class_counts']}")

    if test_result["csv_exists"] and not test_result["missing_required_columns"]:
        print(
            f"Test images present : {test_result['image_exists_count']} | "
            f"missing: {test_result['image_missing_count']}"
        )
        print(f"Test class counts : {test_result['class_counts']}")

    if train_result["missing_required_columns"]:
        print(f"Train CSV missing columns: {train_result['missing_required_columns']}")
    if test_result["missing_required_columns"]:
        print(f"Test CSV missing columns : {test_result['missing_required_columns']}")

    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec}")

    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
