"""Command-line wrapper for single-band motor-imagery EEG preprocessing."""

from src.data.preprocessing import *  # noqa: F401,F403
from src.data.preprocessing import main


if __name__ == "__main__":
    raise SystemExit(main())

