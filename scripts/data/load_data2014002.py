"""Command-line wrapper for downloading BNCI2014-002 MAT files through MOABB."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.download_bnci2014002 import download_bnci2014_002_mat, expected_files, main


if __name__ == "__main__":
    raise SystemExit(main())
