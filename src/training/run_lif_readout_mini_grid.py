"""Run a compact hyperparameter grid for the LIF readout architecture."""

import argparse
import json
import os
import re
import subprocess
import sys
import time


VARIANTS = {
    "official_adam": [],
    "adamw_wd1e4": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
    ],
    "adamw_cosine_wd1e4": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
        "--scheduler",
        "cosine",
    ],
    "soft_aug_no_reverse_adamw_cosine": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
    ],
    "soft_aug_reverse_adamw_cosine": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.25",
    ],
    "soft_aug_no_reverse_label_smooth": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
        "--label_smoothing",
        "0.05",
    ],
    "soft_aug_no_reverse_adam_cosine": [
        "--optimizer",
        "adam",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
    ],
    "soft_aug_no_reverse_adamw_lr3e3": [
        "--lr",
        "0.003",
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
    ],
    "soft_aug_no_reverse_adamw_lr5e3": [
        "--lr",
        "0.005",
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
    ],
    "soft_aug_no_reverse_adamw_wd5e5": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.00005",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
    ],
    "soft_aug_no_reverse_adamw_wd5e4": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0005",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
    ],
    "soft_aug_no_reverse_light_noise": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
        "--aug_gaussian_std_min",
        "0.002",
        "--aug_gaussian_std_max",
        "0.01",
        "--aug_scale_min",
        "0.98",
        "--aug_scale_max",
        "1.02",
        "--aug_shift_max",
        "3",
    ],
    "soft_aug_no_reverse_stronger": [
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--augment_train",
        "--aug_reverse_prob",
        "0.0",
        "--aug_gaussian_std_min",
        "0.01",
        "--aug_gaussian_std_max",
        "0.03",
        "--aug_scale_min",
        "0.9",
        "--aug_scale_max",
        "1.1",
        "--aug_shift_max",
        "10",
    ],
}


def parse_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_summary(log_path):
    result_pattern = re.compile(r"results is: ([0-9.]+)\+-([0-9.]+)")
    json_prefix = "JSON_SUMMARY "
    last_result = None
    json_summary = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(json_prefix):
                json_summary = json.loads(line[len(json_prefix) :])
            match = result_pattern.search(line)
            if match:
                last_result = {
                    "result_mean": float(match.group(1)),
                    "result_std": float(match.group(2)),
                }
    if json_summary is not None:
        return json_summary
    return last_result


def main():
    parser = argparse.ArgumentParser(description="Mini grid for official-loop CUPY_SNN_LIF_READOUT")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--trial_num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--epoch", type=int, default=1500)
    parser.add_argument("--epoch2", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr2", type=float, default=0.0001)
    parser.add_argument("--variants", default="official_adam,adamw_wd1e4,adamw_cosine_wd1e4,soft_aug_no_reverse_adamw_cosine,soft_aug_reverse_adamw_cosine,soft_aug_no_reverse_label_smooth")
    parser.add_argument("--log_dir", default=os.path.join("logs", "training", "lif_readout_grid"))
    parser.add_argument("--out_json", default=os.path.join("benchmarks", "grid", "lif_readout_mini_grid.json"))
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    selected = parse_csv(args.variants)
    results = []
    for variant in selected:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"{variant}_{stamp}.log")
        cmd = [
            args.python,
            "run_official_lif_readout.py",
            "--dataset",
            str(args.dataset),
            "--device",
            str(args.device),
            "--trial_num",
            str(args.trial_num),
            "--seed",
            str(args.seed),
            "--epoch",
            str(args.epoch),
            "--epoch2",
            str(args.epoch2),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--lr2",
            str(args.lr2),
            "--run_name",
            variant,
        ] + VARIANTS[variant]

        print("Running:", " ".join(cmd))
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

        summary = parse_summary(log_path) or {}
        item = {
            "variant": variant,
            "return_code": proc.returncode,
            "log_path": log_path,
            "summary": summary,
        }
        results.append(item)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "params": vars(args),
                    "results": results,
                },
                f,
                indent=2,
            )
        print("Done:", variant, summary)
        if proc.returncode != 0:
            print(f"Variant failed: {variant} code={proc.returncode}")

    print("Saved", args.out_json)


if __name__ == "__main__":
    main()
