"""Train SNN variants on the lab-made motor-imagery dataset."""

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tools.augmentations import EEGAugmentedDataset


EEG_COLUMNS = [f" EXG Channel {i}" for i in range(8)]
LABEL_TO_ID = {
    ("arm", 0): 0,
    ("arm", 1): 1,
    ("leg", 0): 2,
    ("leg", 1): 3,
}
ID_TO_LABEL = {v: f"{k[0]}_{k[1]}" for k, v in LABEL_TO_ID.items()}


parser = argparse.ArgumentParser(description="Train SNN models on lab_made/data_w_labels CSV recordings")
parser.add_argument("--data_root", type=str, default="data/lab_made/data_w_labels")
parser.add_argument("--cache_path", type=str, default="")
parser.add_argument("--type_filter", type=str, default="both", choices=["both", "real", "imagination"])
parser.add_argument("--window_size", type=int, default=500)
parser.add_argument("--stride", type=int, default=250)
parser.add_argument("--min_segment_size", type=int, default=500)
parser.add_argument("--train_subjects", type=str, default="1-12")
parser.add_argument("--val_subjects", type=str, default="13-15")
parser.add_argument("--test_subjects", type=str, default="16-18")
parser.add_argument("--models", type=str, default="CUPY_SNN_LIF_READOUT,CUPY_SNN_LIF_PLIF_LIF_READOUT,CUPY_SNN_SIGNED_LIF_MLP_READOUT")
parser.add_argument("--epochs", type=int, default=35)
parser.add_argument("--patience", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--beta", type=float, default=2.0)
parser.add_argument("--readout_v_threshold", type=float, default=0.2)
parser.add_argument("--readout_input_scale", type=float, default=2.5)
parser.add_argument("--lif_v_threshold", type=float, default=0.5)
parser.add_argument("--lif_input_scale", type=float, default=2.5)
parser.add_argument("--encoder_threshold", type=float, default=0.5)
parser.add_argument("--encoder_scale", type=float, default=1.0)
parser.add_argument("--hidden_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--augment_train", action="store_true")
parser.add_argument("--aug_copies", type=int, default=1)
parser.add_argument("--aug_reverse_prob", type=float, default=0.0)
parser.add_argument("--aug_gaussian_std_min", type=float, default=0.005)
parser.add_argument("--aug_gaussian_std_max", type=float, default=0.02)
parser.add_argument("--aug_scale_min", type=float, default=0.95)
parser.add_argument("--aug_scale_max", type=float, default=1.05)
parser.add_argument("--aug_shift_max", type=int, default=5)
parser.add_argument("--run_name", type=str, default="lab_made_snn")
parser.add_argument("--output_dir", type=str, default="logs/training/lab_made_snn")


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        sig = torch.sigmoid(alpha * x)
        return grad_output * alpha * sig * (1.0 - sig), None


def spike_fn(x, alpha=4.0):
    return SurrogateSpike.apply(x, alpha)


class LIFSequence(nn.Module):
    def __init__(self, threshold=0.5, decay=0.6, learn_decay=False):
        super().__init__()
        self.threshold = float(threshold)
        if learn_decay:
            init = torch.logit(torch.tensor(float(decay)).clamp(1e-4, 1 - 1e-4))
            self.decay_logit = nn.Parameter(init)
        else:
            self.register_buffer("decay_const", torch.tensor(float(decay)))
            self.decay_logit = None

    def decay(self):
        if self.decay_logit is not None:
            return torch.sigmoid(self.decay_logit)
        return self.decay_const

    def forward(self, x):
        # x: [T, N, C]
        v = torch.zeros_like(x[0])
        outputs = []
        decay = self.decay()
        for t in range(x.shape[0]):
            v = decay * v + x[t]
            spike = spike_fn(v - self.threshold)
            v = v * (1.0 - spike.detach())
            outputs.append(spike)
        return torch.stack(outputs, dim=0)


class TorchConvLIFReadout(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_num=4,
        beta=2.0,
        time_step=1000,
        readout_v_threshold=0.2,
        readout_input_scale=2.5,
        lif_v_threshold=0.5,
        lif_input_scale=1.0,
        dropout=0.0,
    ):
        super().__init__()
        channels = max(4, int(beta * in_channels))
        kernel = max(3, time_step // 32)
        if kernel % 2 == 0:
            kernel += 1
        self.encode_c = nn.Conv1d(in_channels, channels, kernel_size=1, bias=False)
        self.encode_t = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=channels,
            bias=False,
        )
        self.bn_t = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lif_input_scale = float(lif_input_scale)
        self.lif = LIFSequence(threshold=lif_v_threshold, decay=0.6, learn_decay=False)
        self.classify = nn.Linear(channels, out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = float(readout_input_scale)
        self.readout = LIFSequence(threshold=readout_v_threshold, decay=0.6, learn_decay=False)

    def forward(self, x):
        x = self.encode_c(x)
        x = self.encode_t(x)
        x = self.drop(self.bn_t(x)).permute(2, 0, 1)
        x = self.lif(x * self.lif_input_scale)
        x = self.classify(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = self.readout(x * self.readout_input_scale)
        return x.mean(0)


class TorchConvPLIFReadout(TorchConvLIFReadout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lif = LIFSequence(threshold=kwargs.get("lif_v_threshold", 0.5), decay=0.6, learn_decay=True)


class SignedSpikeEncoder(nn.Module):
    def __init__(self, threshold=0.5, scale=1.0):
        super().__init__()
        self.threshold = float(threshold)
        self.scale = float(scale)

    def forward(self, x):
        x = x * self.scale
        pos = spike_fn(x - self.threshold)
        neg = spike_fn(-x - self.threshold)
        return torch.cat((pos, neg), dim=-1)


class TorchSignedMLPLIFReadout(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_num=4,
        beta=2.0,
        readout_v_threshold=0.2,
        readout_input_scale=2.5,
        encoder_threshold=0.5,
        encoder_scale=1.0,
        hidden_layers=2,
        dropout=0.1,
        lif_v_threshold=0.5,
        **_,
    ):
        super().__init__()
        channels = max(4, int(beta * in_channels))
        hidden_layers = max(1, int(hidden_layers))
        self.input_norm = nn.BatchNorm1d(in_channels)
        self.encoder = SignedSpikeEncoder(encoder_threshold, encoder_scale)
        self.proj_in = nn.Linear(in_channels * 2, channels, bias=False)
        self.norm_in = nn.BatchNorm1d(channels)
        self.drop_in = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lif_in = LIFSequence(threshold=lif_v_threshold, decay=0.6, learn_decay=False)
        self.projs = nn.ModuleList([nn.Linear(channels, channels, bias=False) for _ in range(hidden_layers - 1)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(hidden_layers - 1)])
        self.drops = nn.ModuleList([nn.Dropout(dropout) if dropout > 0 else nn.Identity() for _ in range(hidden_layers - 1)])
        self.lifs = nn.ModuleList([LIFSequence(threshold=lif_v_threshold, decay=0.6, learn_decay=False) for _ in range(hidden_layers - 1)])
        self.classify = nn.Linear(channels, out_num)
        self.bn_readout = nn.BatchNorm1d(out_num)
        self.readout_input_scale = float(readout_input_scale)
        self.readout = LIFSequence(threshold=readout_v_threshold, decay=0.6, learn_decay=False)

    @staticmethod
    def apply_time_bn(x, bn):
        t, n, c = x.shape
        return bn(x.reshape(t * n, c)).reshape(t, n, c)

    def forward(self, x):
        x = self.input_norm(x).permute(2, 0, 1)
        x = self.encoder(x)
        x = self.proj_in(x)
        x = self.drop_in(self.apply_time_bn(x, self.norm_in))
        x = self.lif_in(x)
        for proj, norm, drop, lif in zip(self.projs, self.norms, self.drops, self.lifs):
            x = proj(x)
            x = drop(self.apply_time_bn(x, norm))
            x = lif(x)
        x = self.classify(x)
        x = self.bn_readout(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = self.readout(x * self.readout_input_scale)
        return x.mean(0)


MODEL_ALIASES = {
    "CUPY_SNN_LIF_READOUT": TorchConvLIFReadout,
    "CUPY_SNN_LIF_PLIF_LIF_READOUT": TorchConvPLIFReadout,
    "CUPY_SNN_SIGNED_LIF_MLP_READOUT": TorchSignedMLPLIFReadout,
    "TORCH_SNN_LIF_READOUT": TorchConvLIFReadout,
    "TORCH_SNN_PLIF_READOUT": TorchConvPLIFReadout,
    "TORCH_SNN_SIGNED_LIF_MLP_READOUT": TorchSignedMLPLIFReadout,
}


def reset_model(net):
    try:
        from tools import functional as sj_functional

        sj_functional.reset_net(net)
        return
    except Exception:
        pass
    for module in net.modules():
        reset = getattr(module, "reset", None)
        if callable(reset):
            reset()


def parse_subjects(spec):
    subjects = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(part))
    return sorted(subjects)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def recording_info(path):
    match = re.match(r"^(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse subject id from {path.name}")
    subject = int(match.group(1))
    return subject


def iter_label_segments(df):
    keys = df[["Type", "Body_part", "action"]].astype(str).agg("|".join, axis=1).to_numpy()
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            yield start, i
            start = i


def make_cache_path(args):
    type_part = args.type_filter
    return Path("data/lab_made") / f"lab_made_windows_{type_part}_w{args.window_size}_s{args.stride}.npz"


def build_window_cache(args):
    root = Path(args.data_root)
    files = sorted(root.iterdir(), key=lambda p: (recording_info(p), p.name))
    xs, ys, subjects, sources = [], [], [], []

    for path in files:
        subject = recording_info(path)
        df = pd.read_csv(path)
        if args.type_filter != "both":
            df = df[df["Type"] == args.type_filter].reset_index(drop=True)
        if df.empty:
            continue

        for start, end in iter_label_segments(df):
            segment = df.iloc[start:end]
            if len(segment) < args.min_segment_size:
                continue
            body_part = str(segment["Body_part"].iloc[0])
            action = int(segment["action"].iloc[0])
            if (body_part, action) not in LABEL_TO_ID:
                continue
            label = LABEL_TO_ID[(body_part, action)]
            values = segment[EEG_COLUMNS].to_numpy(dtype=np.float32).T
            for offset in range(0, values.shape[1] - args.window_size + 1, args.stride):
                window = values[:, offset : offset + args.window_size].copy()
                mean = window.mean(axis=1, keepdims=True)
                std = window.std(axis=1, keepdims=True)
                window = (window - mean) / (std + 1e-6)
                xs.append(window.astype(np.float32))
                ys.append(label)
                subjects.append(subject)
                sources.append(path.name)

    if not xs:
        raise RuntimeError(f"No windows were created from {root}")

    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    subject_arr = np.asarray(subjects, dtype=np.int64)
    source_arr = np.asarray(sources)
    cache_path = Path(args.cache_path) if args.cache_path else make_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=x,
        y=y,
        subject=subject_arr,
        source=source_arr,
        id_to_label=np.asarray([ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]),
    )
    return cache_path


class WindowDataset(Dataset):
    def __init__(self, x, y, indices):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        return self.x[idx], int(self.y[idx])


def load_datasets(args):
    cache_path = Path(args.cache_path) if args.cache_path else make_cache_path(args)
    if not cache_path.exists():
        cache_path = build_window_cache(args)

    data = np.load(cache_path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    subjects = data["subject"].astype(np.int64)

    train_subjects = set(parse_subjects(args.train_subjects))
    val_subjects = set(parse_subjects(args.val_subjects))
    test_subjects = set(parse_subjects(args.test_subjects))
    train_idx = np.where(np.isin(subjects, list(train_subjects)))[0]
    val_idx = np.where(np.isin(subjects, list(val_subjects)))[0]
    test_idx = np.where(np.isin(subjects, list(test_subjects)))[0]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise RuntimeError(
            f"Empty split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    return {
        "cache_path": str(cache_path),
        "x_shape": list(x.shape),
        "label_counts": np.bincount(y, minlength=len(ID_TO_LABEL)).tolist(),
        "splits": {
            "train": sorted(train_subjects),
            "val": sorted(val_subjects),
            "test": sorted(test_subjects),
        },
        "datasets": {
            "train": WindowDataset(x, y, train_idx),
            "val": WindowDataset(x, y, val_idx),
            "test": WindowDataset(x, y, test_idx),
        },
    }


def build_model(model_name, args):
    kwargs = {
        "in_channels": 8,
        "out_num": len(ID_TO_LABEL),
        "time_step": args.window_size,
        "beta": args.beta,
        "readout_v_threshold": args.readout_v_threshold,
        "readout_input_scale": args.readout_input_scale,
    }
    if model_name in {"CUPY_SNN_LIF_PLIF_LIF_READOUT", "CUPY_SNN_SPIKING_CONV_LIF_READOUT"}:
        kwargs.update(
            {
                "lif_v_threshold": args.lif_v_threshold,
                "lif_input_scale": args.lif_input_scale,
            }
        )
    if model_name in {"CUPY_SNN_SIGNED_LIF_MLP_READOUT", "CUPY_SNN_SPIKING_CONV_LIF_READOUT"}:
        kwargs.update(
            {
                "encoder_threshold": args.encoder_threshold,
                "encoder_scale": args.encoder_scale,
                "dropout": args.dropout,
                "lif_v_threshold": args.lif_v_threshold,
            }
        )
    if model_name == "CUPY_SNN_SIGNED_LIF_MLP_READOUT":
        kwargs["hidden_layers"] = args.hidden_layers
    if model_name.startswith("CUPY_"):
        from src import models as cupy_models

        return getattr(cupy_models, model_name)(**kwargs).cuda()
    if model_name not in MODEL_ALIASES:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_ALIASES[model_name](**kwargs).cuda()


def accuracy_for_loader(net, loader, criterion=None):
    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for frame, label in loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            logits = net(frame)
            if criterion is not None:
                total_loss += float(criterion(logits, label).item())
                batches += 1
            correct += (logits.argmax(dim=1) == label).float().sum().item()
            total += label.numel()
            reset_model(net)
    avg_loss = total_loss / max(1, batches)
    return correct / max(1, total), avg_loss


def train_one_model(model_name, datasets, args):
    train_set = datasets["train"]
    if args.augment_train:
        train_set = EEGAugmentedDataset(
            train_set,
            copies_per_sample=args.aug_copies,
            reverse_prob=args.aug_reverse_prob,
            gaussian_std_min=args.aug_gaussian_std_min,
            gaussian_std_max=args.aug_gaussian_std_max,
            scale_min=args.aug_scale_min,
            scale_max=args.aug_scale_max,
            shift_max=args.aug_shift_max,
            seed=args.seed,
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    net = build_model(model_name, args)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{args.run_name}_{model_name}_seed{args.seed}.pth"

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0.0
        batches = 0
        for frame, label in train_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            logits = net(frame)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
            reset_model(net)
        scheduler.step()

        train_acc, _ = accuracy_for_loader(net, train_eval_loader)
        val_acc, val_loss = accuracy_for_loader(net, val_loader, criterion)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / max(1, batches),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )
        print(
            f"{model_name} epoch {epoch + 1:03d}: "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(net.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"{model_name}: early stopping at epoch {epoch + 1}")
                break

    net.load_state_dict(torch.load(ckpt_path))
    test_acc, test_loss = accuracy_for_loader(net, test_loader, criterion)
    return {
        "model": model_name,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
    }


def main():
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these SNN models.")
    torch.cuda.set_device(args.device)
    seed_everything(args.seed)

    data_bundle = load_datasets(args)
    print(
        "Dataset:",
        json.dumps(
            {
                "cache_path": data_bundle["cache_path"],
                "x_shape": data_bundle["x_shape"],
                "label_counts": data_bundle["label_counts"],
                "splits": data_bundle["splits"],
                "labels": ID_TO_LABEL,
            },
            sort_keys=True,
        ),
    )

    results = []
    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"Training {model_name}")
        results.append(train_one_model(model_name, data_bundle["datasets"], args))

    best = max(results, key=lambda item: (item["best_val_acc"], item["test_acc"]))
    summary = {
        "run_name": args.run_name,
        "params": vars(args),
        "data": {
            "cache_path": data_bundle["cache_path"],
            "x_shape": data_bundle["x_shape"],
            "label_counts": data_bundle["label_counts"],
            "splits": data_bundle["splits"],
            "labels": ID_TO_LABEL,
        },
        "results": results,
        "best": best,
    }
    summary_path = Path(args.output_dir) / f"{args.run_name}_seed{args.seed}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("JSON_SUMMARY " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(
        f"Best model: {best['model']} "
        f"val_acc={best['best_val_acc']:.4f} test_acc={best['test_acc']:.4f} "
        f"checkpoint={best['checkpoint']}"
    )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
