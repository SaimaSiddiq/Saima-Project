from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATHS = [
    PROJECT_ROOT / "data/processed/csv/mass_train_mapped.csv",
    PROJECT_ROOT / "data/processed/csv/mass_test_mapped.csv",
]
SEARCH_ROOTS = [PROJECT_ROOT / "datasets", PROJECT_ROOT / "data"]
VALID_SUFFIXES = {".dcm", ".png", ".jpg", ".jpeg"}


def resolve_image_path(row: pd.Series) -> Path | None:
    current = row.get("image_path")
    if isinstance(current, str) and current.strip():
        p = Path(current.strip())
        if p.exists():
            return p.resolve()

    search_terms = []
    rel_path = row.get("image file path")
    if isinstance(rel_path, str) and rel_path.strip():
        rel = Path(rel_path.strip())
        if rel.parts:
            search_terms.extend([rel.parts[0], rel.name, rel.parent.name, rel.parent.parent.name])

    lookup_key = row.get("lookup_key")
    if isinstance(lookup_key, str) and lookup_key.strip():
        search_terms.append(lookup_key.strip())
        search_terms.append(lookup_key.split("_")[-1])

    patient_id = row.get("patient_id")
    if isinstance(patient_id, str) and patient_id.strip():
        search_terms.append(patient_id.strip())

    for term in dict.fromkeys([t for t in search_terms if t]):
        for root in SEARCH_ROOTS:
            if not root.exists():
                continue
            for match in root.rglob(f"*{term}*"):
                if match.is_file() and match.suffix.lower() in VALID_SUFFIXES:
                    return match.resolve()
    return None


for csv_path in CSV_PATHS:
    df = pd.read_csv(csv_path)
    changed = 0
    for idx, row in df.iterrows():
        resolved = resolve_image_path(row)
        if resolved is not None:
            current = row.get("image_path")
            if not isinstance(current, str) or not current.strip() or Path(current.strip()).resolve() != resolved:
                df.at[idx, "image_path"] = str(resolved)
                changed += 1
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path.name}: {changed} rows")

print("Done")
