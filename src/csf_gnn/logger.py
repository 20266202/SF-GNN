import csv
import os
from datetime import datetime

def print_header(title: str) -> None:
    bar = "=" * max(10, len(title) + 8)
    print("\n" + bar)
    print(title)
    print(bar)

def ensure_csv(path: str, header: list[str]) -> None:
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

def append_row(path: str, row: dict, header: list[str]) -> None:
    ensure_csv(path, header)
    # coerce non-serializable values
    safe = {}
    for k in header:
        v = row.get(k, "")
        if isinstance(v, (list, tuple, dict)):
            safe[k] = str(v)
        else:
            safe[k] = v
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(safe)
