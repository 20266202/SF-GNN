"""CSF-GNN refactored utilities.

This package was refactored from an original notebook into a GitHub-friendly layout.
"""

from .datasets import load_dataset
from .train_eval import train_csfgnn, baseline_metrics_logreg
from .models import CSFGNN, CSFConv

__all__ = [
    "load_dataset",
    "train_csfgnn",
    "baseline_metrics_logreg",
    "CSFGNN",
    "CSFConv",
]
