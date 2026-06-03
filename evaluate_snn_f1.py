"""Evaluate saved SNN checkpoints with accuracy and F1 metrics on MI EEG datasets."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tools import functional
from src import models
from src.data import EEGset


DATASETS = ["BNCI2014001", "BNCI2014002", "Weibo2014"]
IN_CHANNELS = [22, 15, 60]
OUT_NUM = [4, 2, 6]
SUBJECT_RANGES = [range(1, 10), range(1, 15), range(1, 11)]


class TrainStatNormalizedDataset(Dataset):
    def __init__(self, base_dataset, mean, std, eps=1e-6):
        self.base_dataset = base_dataset
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.eps = float(eps)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        x = np.asarray(x, dtype=np.float32)
        x = (x - self.mean) / (self.std + self.eps)
        return x.astype(np.float32), y


def train_channel_stats(train_set):
    count = 0
    sum_x = None
    sum_x2 = None
    for x, _ in train_set:
        x = np.asarray(x, dtype=np.float64)
        if sum_x is None:
            sum_x = np.zeros((x.shape[0], 1), dtype=np.float64)
            sum_x2 = np.zeros((x.shape[0], 1), dtype=np.float64)
        sum_x += x.sum(axis=1, keepdims=True)
        sum_x2 += (x * x).sum(axis=1, keepdims=True)
        count += x.shape[1]
    mean = sum_x / max(1, count)
    var = np.maximum(sum_x2 / max(1, count) - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def build_model(args):
    kwargs = {
        "in_channels": IN_CHANNELS[args.dataset],
        "out_num": OUT_NUM[args.dataset],
        "time_step": args.T,
        "beta": args.beta,
    }
    if args.model in {
        "CUPY_SNN_LIF_READOUT",
        "CUPY_SNN_SIGNED_LIF_MLP_READOUT",
        "CUPY_SNN_SPIKING_CONV_LIF_READOUT",
        "CUPY_SNN_LIF_PLIF_LIF_READOUT",
    }:
        kwargs.update(
            {
                "readout_v_threshold": args.readout_v_threshold,
                "readout_adapt_scale": args.readout_adapt_scale,
                "readout_tau_adp_scale": args.readout_tau_adp_scale,
                "readout_input_scale": args.readout_input_scale,
            }
        )
    if args.model in {"CUPY_SNN_SIGNED_LIF_MLP_READOUT", "CUPY_SNN_SPIKING_CONV_LIF_READOUT"}:
        kwargs.update(
            {
                "encoder_threshold": args.encoder_threshold,
                "encoder_scale": args.encoder_scale,
                "dropout": args.dropout,
                "lif_v_threshold": args.lif_v_threshold,
            }
        )
        if args.model == "CUPY_SNN_SIGNED_LIF_MLP_READOUT":
            kwargs["hidden_layers"] = args.hidden_layers
        if args.model == "CUPY_SNN_SPIKING_CONV_LIF_READOUT":
            kwargs["lif_input_scale"] = args.lif_input_scale
    if args.model == "CUPY_SNN_LIF_PLIF_LIF_READOUT":
        kwargs.update(
            {
                "lif_v_threshold": args.lif_v_threshold,
                "lif_input_scale": args.lif_input_scale,
            }
        )
    if args.model == "CUPY_SNN_LIF_READOUT":
        kwargs["dropout"] = args.lif_dropout
    return getattr(models, args.model)(**kwargs).to(args.device_name)


def checkpoint_path(data_path, args, subject_id, seed):
    suffix = ""
    if args.loo:
        suffix += "_loo"
    if args.EA:
        suffix += "_EA"
    if args.run_name:
        name = f"{args.model}_{args.run_name}_id{subject_id}_seed{seed}{suffix}.pth"
    else:
        name = f"{args.model}_id{subject_id}_seed{seed}{suffix}.pth"
    return data_path / name


def update_confusion(confusion, true_labels, pred_labels):
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion[int(true_label), int(pred_label)] += 1


def per_class_metrics(confusion):
    rows = []
    f1_values = []
    recalls = []
    weights = []
    for class_id in range(confusion.shape[0]):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - confusion[class_id, class_id])
        fn = float(confusion[class_id, :].sum() - confusion[class_id, class_id])
        support = int(confusion[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        f1_values.append(f1)
        recalls.append(recall)
        weights.append(support)
    weights = np.asarray(weights, dtype=np.float64)
    f1_values = np.asarray(f1_values, dtype=np.float64)
    return rows, float(np.mean(recalls)), float(np.mean(f1_values)), float(np.average(f1_values, weights=weights))


def build_test_set(data_path, args, subject_id):
    test_set = EEGset(
        root_path=str(data_path) + os.sep,
        pick_id=(subject_id,),
        settup="test",
        T=args.T,
        loo=args.loo,
        all_id=SUBJECT_RANGES[args.dataset],
        EA=args.EA,
        zscore=args.zscore,
    )
    if not args.train_stat_norm:
        return test_set
    train_set = EEGset(
        root_path=str(data_path) + os.sep,
        pick_id=(subject_id,),
        settup="train",
        T=args.T,
        loo=args.loo,
        all_id=SUBJECT_RANGES[args.dataset],
        EA=args.EA,
        zscore=args.zscore,
    )
    stat_mean, stat_std = train_channel_stats(train_set)
    return TrainStatNormalizedDataset(test_set, stat_mean, stat_std, args.train_stat_eps)


def evaluate_one_subject(data_path, args, subject_id, all_confusion):
    model_path = checkpoint_path(data_path, args, subject_id, args.seed)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    net = build_model(args)
    state_dict = torch.load(model_path, map_location=args.device_name)
    net.load_state_dict(state_dict)
    net.eval()

    test_set = build_test_set(data_path, args, subject_id)
    loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    subject_confusion = np.zeros_like(all_confusion)
    subject_correct = 0
    subject_samples = 0
    with torch.no_grad():
        for frame, label in loader:
            frame = frame.to(args.device_name)
            label = label.reshape(-1).to(args.device_name)
            pred = net(frame).argmax(dim=1)
            subject_correct += int((pred == label).sum().item())
            subject_samples += int(label.numel())
            update_confusion(subject_confusion, label.cpu().numpy(), pred.cpu().numpy())
            functional.reset_net(net)

    return {
        "confusion": subject_confusion,
        "correct": subject_correct,
        "samples": subject_samples,
        "checkpoint": str(model_path),
        "accuracy": subject_correct / subject_samples,
    }


def evaluate(args):
    data_path = Path(__file__).resolve().parent / "data" / DATASETS[args.dataset] / args.prep
    subject_ids = [args.subject_id] if args.subject_id else list(SUBJECT_RANGES[args.dataset])
    num_classes = OUT_NUM[args.dataset]

    all_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    subject_confusions = {}
    per_subject = []
    total_correct = 0
    total_samples = 0

    for subject_id in subject_ids:
        subject_result = evaluate_one_subject(data_path, args, subject_id, all_confusion)
        subject_confusion = subject_result["confusion"]
        subject_correct = subject_result["correct"]
        subject_samples = subject_result["samples"]

        all_confusion += subject_confusion
        total_correct += subject_correct
        total_samples += subject_samples
        subject_confusions[str(subject_id)] = subject_confusion.tolist()
        per_subject.append(
            {
                "subject": subject_id,
                "samples": subject_samples,
                "accuracy": subject_correct / subject_samples,
            }
        )

    per_class, balanced_accuracy, macro_f1, weighted_f1 = per_class_metrics(all_confusion)
    result = {
        "dataset": args.dataset,
        "dataset_name": DATASETS[args.dataset],
        "model": args.model,
        "run_name": args.run_name,
        "seed": args.seed,
        "subjects": subject_ids,
        "checkpoint_pattern": checkpoint_path(data_path, args, "{subject}", args.seed).name,
        "total_samples": total_samples,
        "overall_accuracy": total_correct / total_samples,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "per_subject": per_subject,
        "confusion_matrix": all_confusion.tolist(),
        "subject_confusions": subject_confusions,
        "params": vars(args),
    }
    return result


def args_from_params(base_args, params):
    merged = vars(base_args).copy()
    for key, value in params.items():
        if key in merged:
            merged[key] = value
    merged["device_name"] = base_args.device_name
    return argparse.Namespace(**merged)


def selected_subject_args(base_args, selection_payload):
    rows_by_log = {row["log_path"]: row for row in selection_payload.get("results", [])}
    selected = []
    for row in selection_payload["best_by_subject"]:
        full_row = rows_by_log.get(row["log_path"])
        if full_row is None:
            raise KeyError(f"Could not find result row for {row['log_path']}")
        params = full_row.get("summary", {}).get("params", {})
        subject_args = args_from_params(base_args, params)
        selected.append((int(row["subject"]), subject_args, row))
    return selected


def evaluate_selection(args):
    selection_payload = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    selected = selected_subject_args(args, selection_payload)
    if not selected:
        raise ValueError(f"No selected subjects found in {args.selection_json}")

    dataset = selected[0][1].dataset
    data_path = Path(__file__).resolve().parent / "data" / DATASETS[dataset] / selected[0][1].prep
    num_classes = OUT_NUM[dataset]
    all_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    subject_confusions = {}
    per_subject = []
    total_correct = 0
    total_samples = 0

    for subject_id, subject_args, source_row in selected:
        subject_result = evaluate_one_subject(data_path, subject_args, subject_id, all_confusion)
        all_confusion += subject_result["confusion"]
        total_correct += subject_result["correct"]
        total_samples += subject_result["samples"]
        subject_confusions[str(subject_id)] = subject_result["confusion"].tolist()
        per_subject.append(
            {
                "subject": subject_id,
                "samples": subject_result["samples"],
                "accuracy": subject_result["accuracy"],
                "variant": source_row.get("variant"),
                "source_log_path": source_row.get("log_path"),
                "checkpoint": subject_result["checkpoint"],
            }
        )

    per_class, balanced_accuracy, macro_f1, weighted_f1 = per_class_metrics(all_confusion)
    return {
        "dataset": dataset,
        "dataset_name": DATASETS[dataset],
        "selection_json": args.selection_json,
        "subjects": [subject_id for subject_id, _, _ in selected],
        "total_samples": total_samples,
        "overall_accuracy": total_correct / total_samples,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "per_subject": per_subject,
        "confusion_matrix": all_confusion.tolist(),
        "subject_confusions": subject_confusions,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved SNN checkpoints with macro F1.")
    parser.add_argument("--dataset", type=int, default=0)
    parser.add_argument("--model", type=str, default="CUPY_SNN_LIF_READOUT")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--subject_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--prep", type=str, default="250Hz_preprocess_eeg/")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--beta", type=float, default=2)
    parser.add_argument("--loo", action="store_true")
    parser.add_argument("--EA", action="store_true")
    parser.add_argument("--readout_v_threshold", type=float, default=0.2)
    parser.add_argument("--readout_adapt_scale", type=float, default=0.02)
    parser.add_argument("--readout_tau_adp_scale", type=float, default=6.0)
    parser.add_argument("--readout_input_scale", type=float, default=2.5)
    parser.add_argument("--encoder_threshold", type=float, default=0.5)
    parser.add_argument("--encoder_scale", type=float, default=1.0)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lif_dropout", type=float, default=0.0)
    parser.add_argument("--lif_v_threshold", type=float, default=0.5)
    parser.add_argument("--lif_input_scale", type=float, default=2.5)
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--train_stat_norm", action="store_true")
    parser.add_argument("--train_stat_eps", type=float, default=1e-6)
    parser.add_argument("--selection_json", type=str, default="")
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    if args.cpu:
        args.device_name = "cpu"
    else:
        torch.cuda.set_device(args.device)
        args.device_name = f"cuda:{args.device}"
    return args


def main():
    args = parse_args()
    result = evaluate_selection(args) if args.selection_json else evaluate(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_json": str(out_path),
        "overall_accuracy": result["overall_accuracy"],
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
