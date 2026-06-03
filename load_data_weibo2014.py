"""Command-line wrapper for downloading and flattening Weibo2014 MAT files."""

from src.data.download_weibo2014 import download_weibo2014_to_target, main


if __name__ == "__main__":
    raise SystemExit(main())

