"""Download BNCI2014-002 MAT files with MOABB and expose a flat data directory."""

from __future__ import annotations

import os
from pathlib import Path
from shutil import copy2

from moabb.datasets import BNCI2014_002


def expected_files() -> list[str]:
    return [f"S{subject:02d}{part}.mat" for subject in range(1, 15) for part in ("T", "E")]


def download_bnci2014_002_mat(repo_root: Path | None = None) -> tuple[list[str], list[str]]:
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    target_dir = root / "data" / "BNCI2014002"
    target_dir.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    os.chdir(root)
    try:
        dataset = BNCI2014_002()
        copied: set[str] = set()

        for subject in dataset.subject_list:
            paths = dataset.data_path(
                subject=subject,
                path="data/BNCI2014002",
                force_update=False,
                update_path=False,
            )
            for source_name in paths:
                source = Path(source_name)
                if source.suffix.lower() != ".mat":
                    continue
                destination = target_dir / source.name
                if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                    copy2(source, destination)
                copied.add(destination.name)
    finally:
        os.chdir(previous_cwd)

    missing = [name for name in expected_files() if not (target_dir / name).exists()]
    return sorted(copied), missing


def main() -> int:
    copied_files, missing_files = download_bnci2014_002_mat()
    print(f"MAT files copied/found: {len(copied_files)}")
    if missing_files:
        print("Missing files:")
        for file_name in missing_files:
            print(file_name)
    else:
        print("All BNCI2014_002 MAT files are available in data/BNCI2014002")
    return 0

