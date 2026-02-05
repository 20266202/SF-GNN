from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_grid import run  # noqa: E402

if __name__ == "__main__":
    run("income", log_path="experiments_income.csv", seed=42, progress=True)
