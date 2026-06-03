"""Preprocess raw motor-imagery EEG files into subject-wise numpy train splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import mne
import numpy as np
import scipy.io as scio
from moabb.datasets import download as moabb_download
from scipy import signal


@dataclass(frozen=True)
class FilterBand:
    low_hz: float
    high_hz: float
    order: int = 6


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    subjects: range
    extractor: Callable
    default_resample_fs: int


DEFAULT_BAND = FilterBand(8, 30)
FILTERBANK_BANDS = tuple(FilterBand(low, low + 4) for low in range(4, 40, 4))
WEIBO_CHANNELS = tuple(channel for channel in range(62) if channel not in {57, 61})
WEIBO_LABEL_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def load_bnci2014001_test_labels(truth_path):
    mat = scio.loadmat(truth_path)
    if "classlabel" in mat:
        return np.asarray(mat["classlabel"]).reshape(-1) - 1
    if "data" in mat:
        labels = []
        for run_index in range(mat["data"].shape[1]):
            run = mat["data"][0, run_index]
            if run.size == 0:
                continue
            run_labels = run[0, 0]["y"]
            if run_labels.size > 0:
                labels.append(run_labels.reshape(-1))
        if labels:
            return np.concatenate(labels).astype(int) - 1
    raise KeyError(f"Could not find test labels in {truth_path}.")


def preprocessing(data, fs, Fstop1=8, Fstop2=30):
    return bandpass_eeg(data, fs, FilterBand(Fstop1, Fstop2))


def preprocessing2(data, fs, Fstop1=8, Fstop2=26):
    return bandpass_eeg(data, fs, FilterBand(Fstop1, Fstop2, order=5))


def bandpass_eeg(data, fs: int, band: FilterBand, time_axis: int = 1):
    detrended = signal.detrend(data, axis=time_axis, type="linear")
    cutoff = [2.0 * band.low_hz / fs, 2.0 * band.high_hz / fs]
    b_coeff, a_coeff = signal.butter(band.order, cutoff, "bandpass")
    return signal.filtfilt(b_coeff, a_coeff, detrended, axis=time_axis)


def resample_eeg(data, source_fs: int, target_fs: int, time_axis: int = 1):
    if source_fs == target_fs:
        return data, source_fs
    sample_count = int(data.shape[time_axis] / (source_fs / target_fs))
    return signal.resample(data, sample_count, axis=time_axis), target_fs


def _stack_labeled(class_trials: dict[int, list[np.ndarray]], class_order: Iterable[int]):
    data_blocks = []
    label_blocks = []
    for class_id in class_order:
        trials = class_trials[class_id]
        if not trials:
            continue
        data_blocks.append(np.stack(trials, axis=0))
        label_blocks.append(np.full(len(trials), class_id, dtype=int))
    return np.vstack(data_blocks), np.concatenate(label_blocks)


def _split_classwise(class_trials: dict[int, list[np.ndarray]], ratio: float):
    first, second = {}, {}
    for class_id, trials in class_trials.items():
        split_at = int(len(trials) * ratio)
        first[class_id] = trials[:split_at]
        second[class_id] = trials[split_at:]
    return first, second


def _extract_gdf_epochs(raw_path: Path, band: FilterBand, resample_fs: int):
    raw = mne.io.read_raw_gdf(
        raw_path,
        preload=True,
        exclude=["EOG-left", "EOG-central", "EOG-right"],
    )
    fs = int(raw.info["sfreq"])
    eeg = raw.to_data_frame().drop(["time"], axis=1).T.to_numpy()
    eeg = bandpass_eeg(eeg, fs, band)
    eeg, fs = resample_eeg(eeg, fs, resample_fs)
    return eeg, raw.annotations.description, raw.annotations.onset, fs


def get_source_eeg_BNCI2014001(
    person_id,
    current_working_dir,
    train,
    resample_fs=250,
    Fstop1=8,
    Fstop2=30,
):
    source_dir = Path(current_working_dir)
    suffix = "T" if train else "E"
    raw_path = source_dir / f"A{person_id:02d}{suffix}.gdf"
    eeg, event_type, event_position, fs = _extract_gdf_epochs(
        raw_path,
        FilterBand(Fstop1, Fstop2),
        resample_fs,
    )

    if train:
        event_to_class = {"769": 0, "770": 1, "771": 2, "772": 3}
        grouped = {class_id: [] for class_id in event_to_class.values()}
        for event_name, onset in zip(event_type, event_position):
            class_id = event_to_class.get(event_name)
            if class_id is None:
                continue
            start = int(onset * fs)
            grouped[class_id].append(eeg[:, start : start + fs * 4])
        train_trials, validate_trials = _split_classwise(grouped, ratio=5 / 6)
        train_data, train_label = _stack_labeled(train_trials, range(4))
        validate_data, validate_label = _stack_labeled(validate_trials, range(4))
        return train_data, train_label, validate_data, validate_label

    unknown_epochs = []
    for event_name, onset in zip(event_type, event_position):
        if event_name == "783":
            start = int(onset * fs)
            unknown_epochs.append(eeg[:, start : start + fs * 4])
    truth_path = source_dir / f"A{person_id:02d}E.mat"
    return np.stack(unknown_epochs, axis=0), load_bnci2014001_test_labels(truth_path)


def _iter_bnci2014002_runs(mat_data, positions):
    for position in positions:
        run = mat_data[0, position]
        yield {
            "fs": int(run["fs"][0, 0][0, 0]),
            "trial": run["trial"][0, 0],
            "label": run["y"][0, 0] - 1,
            "eeg": run["X"][0, 0].T,
        }


def get_source_eeg_BNCI2014002(
    person_id,
    current_working_dir,
    train,
    resample_fs=128,
    Fstop1=8,
    Fstop2=30,
):
    source_dir = Path(current_working_dir)
    suffix = "T" if train else "E"
    positions = range(5) if train else range(3)
    mat_data = scio.loadmat(source_dir / f"S{person_id:02d}{suffix}.mat")["data"]
    grouped = {0: [], 1: []}

    for run in _iter_bnci2014002_runs(mat_data, positions):
        eeg = preprocessing2(run["eeg"], run["fs"], Fstop1, Fstop2)
        eeg, fs = resample_eeg(eeg, run["fs"], resample_fs)
        trials = run["trial"] / run["fs"] * fs if fs != run["fs"] else run["trial"]
        for trial_index, trial_start in enumerate(trials[0, :]):
            class_id = int(run["label"][0, trial_index])
            if class_id not in grouped:
                continue
            start = int(trial_start) + fs * 4
            grouped[class_id].append(eeg[:, start : start + fs * 4])

    if train:
        train_trials, validate_trials = _split_classwise(grouped, ratio=5 / 6)
        train_data, train_label = _stack_labeled(train_trials, range(2))
        validate_data, validate_label = _stack_labeled(validate_trials, range(2))
        return train_data, train_label, validate_data, validate_label

    return _stack_labeled(grouped, range(2))


def _read_weibo_subject(subject_file: Path, band: FilterBand, resample_fs: int):
    mat = scio.loadmat(subject_file)
    eeg = mat["data"]
    labels = mat["label"]
    fs = 200
    eeg = bandpass_eeg(eeg, fs, band)
    eeg, fs = resample_eeg(eeg, fs, resample_fs)
    return eeg, labels, fs


def _collect_weibo_trials(subject_file: Path, band: FilterBand, resample_fs: int):
    eeg, labels, fs = _read_weibo_subject(subject_file, band, resample_fs)
    grouped = {class_id: [] for class_id in WEIBO_LABEL_TO_INDEX.values()}
    for trial_index in range(eeg.shape[2]):
        class_id = WEIBO_LABEL_TO_INDEX.get(int(labels[trial_index, 0]))
        if class_id is None:
            continue
        start = fs * 3
        stop = fs * 7
        grouped[class_id].append(eeg[WEIBO_CHANNELS, start:stop, trial_index])
    return grouped


def get_source_eeg_from_Weibo2014(
    person_id,
    current_working_dir,
    train,
    resample_fs=128,
    Fstop1=8,
    Fstop2=30,
):
    subject_file = Path(current_working_dir) / f"subject_{person_id}.mat"
    grouped = _collect_weibo_trials(subject_file, FilterBand(Fstop1, Fstop2), resample_fs)
    first_half = {class_id: trials[: len(trials) // 2] for class_id, trials in grouped.items()}
    second_half = {class_id: trials[len(trials) // 2 :] for class_id, trials in grouped.items()}

    if train:
        train_trials, validate_trials = _split_classwise(first_half, ratio=5 / 6)
        train_data, train_label = _stack_labeled(train_trials, range(6))
        validate_data, validate_label = _stack_labeled(validate_trials, range(6))
        return train_data, train_label, validate_data, validate_label

    return _stack_labeled(second_half, range(6))


def read_and_preprocess_for_HighGamma(path_now, resample_fs, current_working_dir):
    local_path = moabb_download.data_dl(
        path_now,
        "SCHIRRMEISTER2017",
        path=current_working_dir,
        force_update=False,
        verbose=None,
    )
    raw = mne.io.read_raw_edf(
        local_path,
        infer_types=True,
        preload=True,
        exclude=["EOG EOGh", "EOG EOGv", "EMG EMG_RH", "EMG EMG_LH", "EMG EMG_RF"],
    )
    selected_channels = [
        "Fp1",
        "Fpz",
        "Fp2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POz",
        "PO4",
        "PO6",
        "PO8",
        "CB1",
        "O1",
        "Oz",
        "O2",
        "CB2",
    ]
    channel_indices = [idx for idx, ch in enumerate(selected_channels) if ch in raw.info["ch_names"]]
    fs = int(raw.info["sfreq"])
    eeg = raw.to_data_frame().drop(["time"], axis=1).T.to_numpy()
    eeg = preprocessing(eeg, fs)
    eeg, fs = resample_eeg(eeg, fs, resample_fs)
    return eeg, raw.annotations.description, raw.annotations.onset, fs, channel_indices


def get_source_eeg_HighGamma(person_id, current_working_dir, resample_fs=250):
    grouped_sessions = []
    for split_name in ("train", "test"):
        url = (
            "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"
            f"/{split_name}/{person_id}.edf"
        )
        eeg, event_type, event_position, fs, channel_indices = read_and_preprocess_for_HighGamma(
            url,
            resample_fs,
            current_working_dir,
        )
        grouped = {0: [], 1: [], 2: [], 3: []}
        event_to_class = {"left_hand": 0, "right_hand": 1, "feet": 2, "rest": 3}
        for event_name, onset in zip(event_type, event_position):
            class_id = event_to_class.get(event_name)
            if class_id is None:
                continue
            start = int(onset * fs)
            grouped[class_id].append(eeg[channel_indices, start : start + fs * 4])
        grouped_sessions.append(grouped)

    train_groups = {}
    test_groups = {}
    for class_id in range(4):
        session_trials = [session[class_id] for session in grouped_sessions]
        train_groups[class_id] = []
        test_groups[class_id] = []
        for trials in session_trials:
            split_at = int(len(trials) * 0.7)
            train_groups[class_id].extend(trials[:split_at])
            test_groups[class_id].extend(trials[split_at:])

    train_data, train_label = _stack_labeled(train_groups, range(4))
    test_data, test_label = _stack_labeled(test_groups, range(4))
    return train_data, train_label, test_data, test_label


def Zsolve(R):
    values, vectors = np.linalg.eigh(R)
    values = np.maximum(values, 1e-12)
    return np.real(vectors @ np.diag(values ** -0.5) @ vectors.T)


def _align_trials_3d(trials):
    covariance = np.zeros((trials.shape[1], trials.shape[1]), dtype=np.float64)
    for trial in trials:
        covariance += trial @ trial.T
    whitening = Zsolve(covariance / trials.shape[0])
    return np.asarray([whitening @ trial for trial in trials])


def EA(X):
    if X.ndim == 3:
        return _align_trials_3d(X)
    if X.ndim == 4:
        return np.stack([_align_trials_3d(X[..., band]) for band in range(X.shape[-1])], axis=-1)
    raise ValueError(f"EA expects a 3D or 4D EEG array, got shape {X.shape}")


def _save_array_pair(split_dir: Path, person_id: int, data, labels) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"data_id{person_id}.npy", data)
    np.save(split_dir / f"label_id{person_id}.npy", labels)


def save_data_label(work_path, person_id, train_data, train_label, validate_data, validate_label, test_data, test_label):
    output_dir = Path(work_path)
    _save_array_pair(output_dir / "train", person_id, train_data, train_label)
    _save_array_pair(output_dir / "validate", person_id, validate_data, validate_label)
    _save_array_pair(output_dir / "test", person_id, test_data, test_label)

    train_validate = np.concatenate((train_data, validate_data), axis=0)
    aligned_train_validate = EA(train_validate)
    aligned_train = aligned_train_validate[: train_data.shape[0]]
    aligned_validate = aligned_train_validate[train_data.shape[0] :]
    aligned_test = EA(test_data)

    _save_array_pair(output_dir / "train" / "EA", person_id, aligned_train, train_label)
    _save_array_pair(output_dir / "validate" / "EA", person_id, aligned_validate, validate_label)
    _save_array_pair(output_dir / "test" / "EA", person_id, aligned_test, test_label)


def save_data_label2(work_path, person_id, train_data, train_label, test_data, test_label):
    output_dir = Path(work_path)
    _save_array_pair(output_dir / "train", person_id, train_data, train_label)
    _save_array_pair(output_dir / "test", person_id, test_data, test_label)


DATASETS = {
    0: DatasetSpec("BNCI2014001", range(1, 10), get_source_eeg_BNCI2014001, 250),
    1: DatasetSpec("BNCI2014002", range(1, 15), get_source_eeg_BNCI2014002, 128),
    2: DatasetSpec("Weibo2014", range(1, 11), get_source_eeg_from_Weibo2014, 250),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_root(raw_dir: Path, resample_fs: int, output_dir: str | None, filterbank: bool):
    if output_dir:
        root = Path(output_dir).resolve() / f"{resample_fs}Hz_preprocess_eeg"
    else:
        root = raw_dir / f"{resample_fs}Hz_preprocess_eeg"
    return root / "filterbank" if filterbank else root


def _preprocess_single_band(spec: DatasetSpec, raw_dir: Path, output_dir: Path, subject_id: int, resample_fs: int):
    train_data, train_label, validate_data, validate_label = spec.extractor(
        subject_id,
        raw_dir,
        train=True,
        resample_fs=resample_fs,
    )
    test_data, test_label = spec.extractor(
        subject_id,
        raw_dir,
        train=False,
        resample_fs=resample_fs,
    )
    save_data_label(output_dir, subject_id, train_data, train_label, validate_data, validate_label, test_data, test_label)


def _preprocess_filterbank(spec: DatasetSpec, raw_dir: Path, output_dir: Path, subject_id: int, resample_fs: int):
    train_bands, validate_bands, test_bands = [], [], []
    for band in FILTERBANK_BANDS:
        train_data, train_label, validate_data, validate_label = spec.extractor(
            subject_id,
            raw_dir,
            train=True,
            resample_fs=resample_fs,
            Fstop1=band.low_hz,
            Fstop2=band.high_hz,
        )
        test_data, test_label = spec.extractor(
            subject_id,
            raw_dir,
            train=False,
            resample_fs=resample_fs,
            Fstop1=band.low_hz,
            Fstop2=band.high_hz,
        )
        train_bands.append(train_data)
        validate_bands.append(validate_data)
        test_bands.append(test_data)

    save_data_label(
        output_dir,
        subject_id,
        np.stack(train_bands, axis=-1),
        train_label,
        np.stack(validate_bands, axis=-1),
        validate_label,
        np.stack(test_bands, axis=-1),
        test_label,
    )


def preprocess_dataset(dataset_id: int, resample_fs: int | None = None, output_dir: str | None = None, filterbank=False):
    if dataset_id not in DATASETS:
        raise ValueError(f"Unsupported dataset id {dataset_id}. Available ids: {sorted(DATASETS)}")

    spec = DATASETS[dataset_id]
    target_fs = spec.default_resample_fs if resample_fs is None else resample_fs
    raw_dir = _repo_root() / "data" / spec.name
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"{spec.name} directory does not exist: {raw_dir}")

    processed_dir = _output_root(raw_dir, target_fs, output_dir, filterbank)
    processed_dir.mkdir(parents=True, exist_ok=True)

    runner = _preprocess_filterbank if filterbank else _preprocess_single_band
    for subject_id in spec.subjects:
        runner(spec, raw_dir, processed_dir, subject_id, target_fs)

    return processed_dir


def build_parser(filterbank: bool = False):
    parser = argparse.ArgumentParser(description="MI EEG preprocessing")
    parser.add_argument("--dataset", type=int, default=2, help="Dataset id: 0=BNCI2014001, 1=BNCI2014002, 2=Weibo2014")
    parser.add_argument("--resample_fs", type=int, default=None, help="Target sampling rate")
    parser.add_argument("--output_dir", type=str, default="", help="Optional root directory for processed arrays")
    parser.set_defaults(filterbank=filterbank)
    return parser


def main(argv=None) -> int:
    args = build_parser(filterbank=False).parse_args(argv)
    output_dir = preprocess_dataset(args.dataset, args.resample_fs, args.output_dir or None, filterbank=False)
    print(f"Saved preprocessed EEG arrays to {output_dir}")
    return 0


def main_filterbank(argv=None) -> int:
    args = build_parser(filterbank=True).parse_args(argv)
    output_dir = preprocess_dataset(args.dataset, args.resample_fs, args.output_dir or None, filterbank=True)
    print(f"Saved filterbank EEG arrays to {output_dir}")
    return 0
