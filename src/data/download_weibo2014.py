"""Download Weibo2014 MAT files with MOABB and normalize the subject file layout."""

from __future__ import annotations

import os
from pathlib import Path
from shutil import copy2
from typing import Iterable

from moabb.datasets import Weibo2014


def _subject_files(base_dir: Path, subjects: Iterable[int]) -> list[Path]:
    return [base_dir / f"subject_{subject}.mat" for subject in subjects]


def _link_or_copy(source: Path, target_dir: Path) -> None:
    destination = target_dir / source.name
    if destination.exists() and destination.stat().st_size != source.stat().st_size:
        destination.unlink()
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        copy2(source, destination)


def download_weibo2014_to_target(repo_root: Path | None = None) -> tuple[list[str], int, list[str]]:
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    target_dir = root / "data" / "Weibo2014"
    target_dir.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    os.chdir(root)
    try:
        dataset = Weibo2014()
        moabb_dataset_dir = target_dir / "MNE-weibo-2014"
        linked: set[str] = set()
        repaired_groups: list[str] = []

        subject_groups = {
            1: [1, 2, 3, 4],
            5: [5, 6, 7],
            8: [8, 9, 10],
        }
        for anchor_subject, group in subject_groups.items():
            group_files = _subject_files(moabb_dataset_dir, group)
            exists = [path.exists() for path in group_files]
            if all(exists):
                continue
            if any(exists):
                for file_path in group_files:
                    if file_path.exists():
                        file_path.unlink()
                repaired_groups.append(f"{group[0]}-{group[-1]}")

            dataset.data_path(
                subject=anchor_subject,
                path="data/Weibo2014",
                force_update=False,
                update_path=False,
            )

        for nested_mat in moabb_dataset_dir.glob("subject_*.mat"):
            _link_or_copy(nested_mat, target_dir)
            linked.add(nested_mat.name)
    finally:
        os.chdir(previous_cwd)

    return sorted(linked), len(dataset.subject_list), repaired_groups


def main() -> int:
    copied_files, subjects_count, repaired = download_weibo2014_to_target()
    print(f"Subjects in Weibo2014: {subjects_count}")
    if repaired:
        print("Repaired partial subject groups:", ", ".join(repaired))
    print(f"MAT files copied/found in target: {len(copied_files)}")
    for name in copied_files:
        print(name)
    return 0

