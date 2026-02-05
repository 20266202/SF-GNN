from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


def ensure_csv(path: str, header_cols: List[str]) -> None:
    """Create an empty CSV with the given header if it doesn't exist."""
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        pd.DataFrame(columns=header_cols).to_csv(path, index=False)


def append_row(path: str, row_dict: Dict, header_cols: List[str]) -> None:
    """Append a single row to CSV, respecting a fixed column order."""
    df = pd.DataFrame([[row_dict.get(col, None) for col in header_cols]], columns=header_cols)
    df.to_csv(path, mode="a", header=False, index=False)
