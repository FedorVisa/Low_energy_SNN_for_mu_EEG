"""Train and evaluate the Norse latency-coded SNN baseline."""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from tools.augmentations import EEGAugmentedDataset
from src.data import EEGset
from src.models.norse_models import NORSE_LATENCY_CONV_LIF_READOUT


DATASET_FOLDERS = ["BNCI2014001", "BNCI2014002", "Weibo2014"]
DATASET_CHANNELS = [22, 15, 60]
DATASET_CLASSES = [4, 2, 6]
DATASET_SUBJECTS = [range(1, 10), range(1, 15), range(1, 11)]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SegmentedEEGSet(Dataset):
    def __init__(self, base_set, segment_len=250):
        self.base_set = base_set
        self.segment_len = int(segment_len)
        first_x, _ = base_set[0]
        self.segments_per_trial = max(1, first_x.shape[-1] // self.segment_len)

    def __len__(self):
        return len(self.base_set) * self.segments_per_trial

    def __getitem__(self, index):
        trial_index = index // self.segments_per_trial
        segment_index = index % self.segments_per_trial
        x, y = self.base_set[trial_index]
        start = segment_index * self.segment_len
        end = start + self.segment_len
        return x[:, start:end].astype(np.float32), y


def seed_torch(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_path(args):
    return PROJECT_ROOT / "data" / DATASET_FOLDERS[args.dataset] / args.prep


def build_base_set(data_path, args, subject_id, setup):
    return EEGset(
        root_path=str(data_path) + os.sep,
        pick_id=(subject_id,),
        settup=setup,
        T=args.trial_time_step,
        loo=args.loo,
        all_id=DATASET_SUBJECTS[args.dataset],
        EA=args.EA,
        zscore=args.zscore,
    )


def maybe_augment(train_set, args, seed):
    if not args.augment_train:
        return train_set
    return EEGAugmentedDataset(
        train_set,
        copies_per_sample=args.aug_copies,
        reverse_prob=args.aug_reverse_prob,
        gaussian_std_min=args.aug_gaussian_std_min,
        gaussian_std_max=args.aug_gaussian_std_max,
        scale_min=args.aug_scale_min,
        scale_max=args.aug_scale_max,
        shift_max=args.aug_shift_max,
        seed=seed,
    )


def build_model(args):
    return NORSE_LATENCY_CONV_LIF_READOUT(
        in_channels=DATASET_CHANNELS[args.dataset],
        out_num=DATASET_CLASSES[args.dataset],
        beta=args.beta,
        time_step=args.segment_len,
        latency_topk=args.latency_topk,
        latency_window=args.latency_window,
        lif_v_threshold=args.lif_v_threshold,
        readout_v_threshold=args.readout_v_threshold,
        lif_input_scale=args.lif_input_scale,
        readout_input_scale=args.readout_input_scale,
        tau_mem_ms=args.tau_mem_ms,
        tau_syn_ms=args.tau_syn_ms,
        dropout=args.dropout,
    )


def build_optimizer(parameters, lr, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(parameters, lr=lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, epochs, args):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    return None


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class EarlyStopping:
    def __init__(self, patience, path):
        self.patience = int(patience)
        self.path = path
        self.best_acc = -1.0
        self.counter = 0
        self.early_stop = False

    def __call__(self, acc, model):
        if acc <= self.best_acc:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
            return
        self.best_acc = acc
        self.counter = 0
        torch.save(model.state_dict(), self.path)


def logits_for_trials(model, frames, args, device):
    batch, channels, trial_len = frames.shape
    seg_count = max(1, trial_len // args.segment_len)
    frames = frames[:, :, : seg_count * args.segment_len]
    segments = frames.reshape(batch, channels, seg_count, args.segment_len)
    segments = segments.permute(0, 2, 1, 3).reshape(batch * seg_count, channels, args.segment_len)
    logits = model(segments.to(device))
    return logits.reshape(batch, seg_count, -1).mean(dim=1)


def evaluate_trials(model, base_set, args, device, loss_fn=None):
    loader = DataLoader(base_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for frames, labels in loader:
            labels = labels.reshape(-1).long().to(device)
            logits = logits_for_trials(model, frames.float(), args, device)
            if loss_fn is not None:
                total_loss += float(loss_fn(logits, labels).item())
                batches += 1
            correct += (logits.argmax(dim=1) == labels).float().sum().item()
            total += labels.numel()
    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, batches)
    return acc, avg_loss


def train_stage_one(model, train_set, validate_base, model_path, args, device, loss_fn):
    train_loader = DataLoader(
        SegmentedEEGSet(train_set, segment_len=args.segment_len),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    optimizer = build_optimizer(model.parameters(), args.lr, args)
    scheduler = build_scheduler(optimizer, args.epoch, args)
    early_stop = EarlyStopping(args.patience, model_path)
    last_train_loss = 0.0

    for epoch in range(args.epoch):
        model.train()
        loss_sum = 0.0
        batches = 0
        for frames, labels in train_loader:
            frames = frames.float().to(device)
            labels = labels.reshape(-1).long().to(device)
            logits = model(frames)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
        last_train_loss = loss_sum / max(1, batches)
        if scheduler is not None:
            scheduler.step()

        val_acc, val_loss = evaluate_trials(model, validate_base, args, device, loss_fn)
        print(
            f"epoch={epoch + 1:03d} train_loss={last_train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={100.0 * val_acc:.2f}",
            flush=True,
        )
        early_stop(val_acc, model)
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch + 1}", flush=True)
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    return last_train_loss


def train_stage_two(model, train_set, validate_set, model_path, train_loss, args, device, loss_fn):
    combined = ConcatDataset(
        [
            SegmentedEEGSet(train_set, segment_len=args.segment_len),
            SegmentedEEGSet(validate_set, segment_len=args.segment_len),
        ]
    )
    loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer = build_optimizer(model.parameters(), args.lr2, args)
    scheduler = build_scheduler(optimizer, args.epoch2, args)

    for epoch in range(args.epoch2):
        model.train()
        loss_sum = 0.0
        batches = 0
        for frames, labels in loader:
            frames = frames.float().to(device)
            labels = labels.reshape(-1).long().to(device)
            logits = model(frames)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
        loss_value = loss_sum / max(1, batches)
        if scheduler is not None:
            scheduler.step()
        torch.save(model.state_dict(), model_path)
        if loss_value < train_loss:
            print(f"current loss < stage 1 train loss at epoch {epoch + 1}", flush=True)
            break

    model.load_state_dict(torch.load(model_path, map_location=device))


def parse_cupy_reference():
    candidates = []
    grid_path = Path("benchmarks/results/grid/big_snn_experiments_seed23.json")
    if grid_path.exists():
        with grid_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for result in payload.get("results", []):
            summary = result.get("summary") or {}
            if result.get("model") == "CUPY_SNN_LIF_READOUT" and "result_mean" in summary:
                candidates.append(
                    {
                        "source": str(grid_path),
                        "run_name": result.get("name"),
                        "mean_accuracy_percent": float(summary["result_mean"]),
                        "std_percent": float(summary.get("result_std", 0.0)),
                    }
                )

    log_path = Path("logs/training/big_snn_experiments/max_lif_readout_seed23_adamw_lr3e3_20260513_193609.log")
    if log_path.exists():
        summary = None
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("JSON_SUMMARY "):
                    summary = json.loads(line[len("JSON_SUMMARY ") :])
        if summary:
            candidates.append(
                {
                    "source": str(log_path),
                    "run_name": summary.get("run_name"),
                    "mean_accuracy_percent": float(summary["result_mean"]),
                    "std_percent": float(summary.get("result_std", 0.0)),
                }
            )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["mean_accuracy_percent"])


def read_cupy_energy_reference():
    path = Path("benchmarks/results/model_efficiency/bnci2014_001_params_power_metrics.csv")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == "CUPY_SNN_LIF_READOUT":
                return {
                    "source": str(path),
                    "parameters": int(float(row["parameters"])),
                    "peak_power_w": float(row["peak_power_w"]),
                    "mean_power_w": float(row["mean_power_w"]),
                    "net_peak_power_w": float(row["net_peak_power_w"]),
                    "elapsed_ms_per_batch": float(row["elapsed_ms_per_batch"]),
                    "accuracy_percent_in_file": float(row["accuracy_percent"]),
                }
    return None


def query_power_w(device):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return float(output.strip().splitlines()[0])


class PowerMonitor:
    def __init__(self, device, interval_s):
        self.device = device
        self.interval_s = float(interval_s)
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter(), query_power_w(self.device)))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_power_w(self):
        return max((power for _, power in self.samples), default=math.nan)

    @property
    def mean_power_w(self):
        if not self.samples:
            return math.nan
        return sum(power for _, power in self.samples) / len(self.samples)


def profile_power(model, base_set, args, device):
    if device.type != "cuda" or args.skip_power:
        return None
    try:
        _ = query_power_w(args.device)
    except Exception as exc:
        return {"error": f"Cannot query nvidia-smi power: {exc}"}

    loader = DataLoader(base_set, batch_size=args.power_batch_size, shuffle=False, drop_last=False)
    frames, _ = next(iter(loader))
    frames = frames.float().to(device)
    model.eval()

    time.sleep(0.75)
    idle_samples = []
    for _ in range(5):
        idle_samples.append(query_power_w(args.device))
        time.sleep(args.power_sample_interval)
    idle_power_w = sum(idle_samples) / len(idle_samples)

    with torch.inference_mode():
        for _ in range(args.power_warmup):
            _ = logits_for_trials(model, frames, args, device)
            torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iterations = 0
        wall_start = time.perf_counter()
        with PowerMonitor(args.device, args.power_sample_interval) as monitor:
            torch.cuda.synchronize(device)
            start.record()
            while True:
                for _ in range(args.power_repeat):
                    _ = logits_for_trials(model, frames, args, device)
                iterations += args.power_repeat
                torch.cuda.synchronize(device)
                if time.perf_counter() - wall_start >= args.power_min_seconds:
                    break
            end.record()
            torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    return {
        "parameters": count_parameters(model),
        "batch_size": int(frames.shape[0]),
        "idle_power_w": idle_power_w,
        "peak_power_w": monitor.peak_power_w,
        "mean_power_w": monitor.mean_power_w,
        "net_peak_power_w": monitor.peak_power_w - idle_power_w,
        "elapsed_ms_total": elapsed_ms,
        "elapsed_ms_per_batch": elapsed_ms / max(1, iterations),
        "iterations": iterations,
        "power_samples": len(monitor.samples),
    }


def save_summary(summary, out_json):
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Norse latency-coded SNN for BNCI2014 experiments")
    parser.add_argument("--dataset", type=int, default=0)
    parser.add_argument("--prep", default="250Hz_preprocess_eeg")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--trial_num", type=int, default=1)
    parser.add_argument("--subject_id", type=int, default=0)
    parser.add_argument("--trial_time_step", type=int, default=1000)
    parser.add_argument("--segment_len", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--epoch2", type=int, default=15)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr2", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--latency_topk", type=int, default=16)
    parser.add_argument("--latency_window", type=int, default=10)
    parser.add_argument("--lif_v_threshold", type=float, default=0.5)
    parser.add_argument("--readout_v_threshold", type=float, default=0.2)
    parser.add_argument("--lif_input_scale", type=float, default=2.5)
    parser.add_argument("--readout_input_scale", type=float, default=2.5)
    parser.add_argument("--tau_mem_ms", type=float, default=20.0)
    parser.add_argument("--tau_syn_ms", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--augment_train", action="store_true")
    parser.add_argument("--aug_copies", type=int, default=1)
    parser.add_argument("--aug_reverse_prob", type=float, default=0.0)
    parser.add_argument("--aug_gaussian_std_min", type=float, default=0.005)
    parser.add_argument("--aug_gaussian_std_max", type=float, default=0.02)
    parser.add_argument("--aug_scale_min", type=float, default=0.95)
    parser.add_argument("--aug_scale_max", type=float, default=1.05)
    parser.add_argument("--aug_shift_max", type=int, default=5)
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--loo", action="store_true")
    parser.add_argument("--EA", action="store_true")
    parser.add_argument("--run_name", default="norse_latency_conv_lif_readout")
    parser.add_argument("--log_dir", default=os.path.join("logs", "training", "norse_latency_snn"))
    parser.add_argument("--out_json", default=os.path.join("benchmarks", "grid", "norse_latency_bnci2014_comparison.json"))
    parser.add_argument("--skip_power", action="store_true")
    parser.add_argument("--power_batch_size", type=int, default=128)
    parser.add_argument("--power_warmup", type=int, default=5)
    parser.add_argument("--power_repeat", type=int, default=20)
    parser.add_argument("--power_min_seconds", type=float, default=5.0)
    parser.add_argument("--power_sample_interval", type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset != 0:
        raise ValueError("This methodology comparison is configured for BNCI2014_001 / BCI IV 2a.")

    data_path = build_data_path(args)
    if not data_path.exists():
        raise RuntimeError(f"Missing data path: {data_path}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(args.device)
    else:
        device = torch.device("cpu")
        print("CUDA is not available; running Norse experiment on CPU.", flush=True)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    print(vars(args), flush=True)

    subject_ids = list(DATASET_SUBJECTS[args.dataset])
    if args.subject_id:
        subject_ids = [args.subject_id]

    records = []
    trial_means = []
    last_model = None
    last_test_base = None
    for trial_index in range(args.trial_num):
        seed_value = args.seed + trial_index
        seed_torch(seed_value)
        subject_accs = []
        for subject_id in subject_ids:
            model_path = Path(args.log_dir) / f"{args.run_name}_id{subject_id}_seed{seed_value}.pth"
            train_base = build_base_set(data_path, args, subject_id, "train")
            validate_base = build_base_set(data_path, args, subject_id, "validate")
            test_base = build_base_set(data_path, args, subject_id, "test")
            train_set = maybe_augment(train_base, args, seed_value)

            model = build_model(args).to(device)
            if trial_index == 0 and subject_id == subject_ids[0]:
                print(model, flush=True)
                print(f"trainable_parameters={count_parameters(model)}", flush=True)

            loss_fn = nn.CrossEntropyLoss().to(device)
            print(f"{args.run_name}_trial{trial_index}_id{subject_id}", flush=True)
            train_loss = train_stage_one(model, train_set, validate_base, model_path, args, device, loss_fn)
            train_stage_two(model, train_set, validate_base, model_path, train_loss, args, device, loss_fn)
            test_acc, test_loss = evaluate_trials(model, test_base, args, device, loss_fn)
            accuracy_percent = round(100.0 * test_acc, 4)
            print(f"the test accuracy is {test_acc:.4f} ({accuracy_percent:.2f}%)", flush=True)
            records.append(
                {
                    "seed": seed_value,
                    "subject": subject_id,
                    "accuracy": accuracy_percent,
                    "test_loss": test_loss,
                }
            )
            subject_accs.append(accuracy_percent)
            last_model = model
            last_test_base = test_base

        trial_mean = float(np.mean(subject_accs))
        trial_means.append(trial_mean)
        print(f"seed{seed_value} mean is {trial_mean}", flush=True)

    result_mean = float(np.mean(trial_means))
    result_std = float(math.sqrt(np.var(trial_means)))
    cupy_ref = parse_cupy_reference()
    cupy_energy = read_cupy_energy_reference()
    norse_power = profile_power(last_model, last_test_base, args, device) if last_model is not None else None

    comparison = {}
    if cupy_ref is not None:
        comparison["accuracy_delta_points_vs_best_cupy"] = result_mean - cupy_ref["mean_accuracy_percent"]
    if cupy_energy is not None and isinstance(norse_power, dict) and "elapsed_ms_per_batch" in norse_power:
        comparison["norse_peak_power_delta_w_vs_cupy"] = norse_power["peak_power_w"] - cupy_energy["peak_power_w"]
        comparison["norse_batch_latency_delta_ms_vs_cupy"] = (
            norse_power["elapsed_ms_per_batch"] - cupy_energy["elapsed_ms_per_batch"]
        )
        comparison["norse_params_delta_vs_cupy"] = norse_power["parameters"] - cupy_energy["parameters"]

    summary = {
        "run_name": args.run_name,
        "model": "NORSE_LATENCY_CONV_LIF_READOUT",
        "result_mean": result_mean,
        "result_std": result_std,
        "trial_means": trial_means,
        "records": records,
        "params": vars(args),
        "cupy_lif_readout_reference": cupy_ref,
        "cupy_lif_readout_energy_reference": cupy_energy,
        "norse_energy": norse_power,
        "comparison": comparison,
    }
    save_summary(summary, args.out_json)
    print("JSON_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"results is: {result_mean}+-{result_std}", flush=True)
    print(f"Saved: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
