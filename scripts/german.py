import os, sys

# Allow `python scripts/german.py` from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run_grid import run  # run_grid.py is in the same folder as this script

if __name__ == "__main__":
    run("german")
