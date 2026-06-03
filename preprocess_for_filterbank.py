"""Command-line wrapper for filterbank motor-imagery EEG preprocessing."""

from src.data.preprocessing import *  # noqa: F401,F403
from src.data.preprocessing import main_filterbank


if __name__ == "__main__":
    raise SystemExit(main_filterbank())

