"""Launch larger SNN experiment batches with predefined settings."""

import argparse
import json
import os
import re
import subprocess
import sys
import time


def parse_summary(log_path):
    json_prefix = "JSON_SUMMARY "
    result_pattern = re.compile(r"results is: ([0-9.]+)\+-([0-9.]+)")
    summary = None
    fallback = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(json_prefix):
                summary = json.loads(line[len(json_prefix):])
            match = result_pattern.search(line)
            if match:
                fallback = {
                    "result_mean": float(match.group(1)),
                    "result_std": float(match.group(2)),
                }
    return summary or fallback or {}


def run_one(args, spec):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = spec["name"]
    log_path = os.path.join(args.log_dir, f"{run_name}_{stamp}.log")
    cmd = [
        args.python,
        "-m",
        "src.training.run_official_lif_readout",
        "--dataset", str(args.dataset),
        "--device", str(args.device),
        "--model", spec["model"],
        "--trial_num", str(spec.get("trial_num", args.tune_trial_num)),
        "--seed", str(args.seed),
        "--epoch", str(spec.get("epoch", args.tune_epoch)),
        "--epoch2", str(spec.get("epoch2", args.tune_epoch2)),
        "--patience", str(spec.get("patience", args.patience)),
        "--stage2_select", str(spec.get("stage2_select", args.stage2_select)),
        "--stage2_patience", str(spec.get("stage2_patience", args.stage2_patience)),
        "--batch_size", str(spec.get("batch_size", args.batch_size)),
        "--lr", str(spec.get("lr", args.lr)),
        "--lr2", str(spec.get("lr2", args.lr2)),
        "--run_name", run_name,
    ]
    for key, value in spec.get("flags", {}).items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    started_at = time.time()
    print("RUN", run_name, " ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    elapsed_sec = round(time.time() - started_at, 2)
    summary = parse_summary(log_path)
    return {
        "name": run_name,
        "model": spec["model"],
        "return_code": proc.returncode,
        "elapsed_sec": elapsed_sec,
        "log_path": log_path,
        "summary": summary,
        "command": cmd,
    }


def build_specs(args):
    common_best = {
        "optimizer": "adamw",
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "augment_train": True,
        "aug_reverse_prob": 0.0,
    }
    soft_aug = {
        "augment_train": True,
        "aug_reverse_prob": 0.0,
        "aug_gaussian_std_min": 0.005,
        "aug_gaussian_std_max": 0.02,
        "aug_scale_min": 0.95,
        "aug_scale_max": 1.05,
        "aug_shift_max": 5,
    }

    specs = [
        {
            "name": args.max_run_name,
            "model": "CUPY_SNN_LIF_READOUT",
            "trial_num": args.max_trial_num,
            "epoch": args.max_epoch,
            "epoch2": args.max_epoch2,
            "patience": args.max_patience,
            "lr": 0.003,
            "lr2": 0.0001,
            "flags": common_best,
        }
    ]

    if args.skip_tune:
        return specs

    signed_base = {
        **common_best,
        **soft_aug,
    }
    signed_grid = [
        ("signed_mlp_thr030_scale10_layers2_do00", {"encoder_threshold": 0.3, "encoder_scale": 1.0, "hidden_layers": 2, "dropout": 0.0, "lif_v_threshold": 0.5}),
        ("signed_mlp_thr030_scale15_layers2_do00", {"encoder_threshold": 0.3, "encoder_scale": 1.5, "hidden_layers": 2, "dropout": 0.0, "lif_v_threshold": 0.5}),
        ("signed_mlp_thr020_scale15_layers2_do00", {"encoder_threshold": 0.2, "encoder_scale": 1.5, "hidden_layers": 2, "dropout": 0.0, "lif_v_threshold": 0.4}),
        ("signed_mlp_thr020_scale20_layers3_do05", {"encoder_threshold": 0.2, "encoder_scale": 2.0, "hidden_layers": 3, "dropout": 0.05, "lif_v_threshold": 0.4}),
        ("signed_mlp_thr010_scale20_layers3_do00", {"encoder_threshold": 0.1, "encoder_scale": 2.0, "hidden_layers": 3, "dropout": 0.0, "lif_v_threshold": 0.3}),
        ("signed_mlp_thr050_scale20_layers2_do10", {"encoder_threshold": 0.5, "encoder_scale": 2.0, "hidden_layers": 2, "dropout": 0.1, "lif_v_threshold": 0.5}),
    ]
    for name, flags in signed_grid:
        specs.append(
            {
                "name": name,
                "model": "CUPY_SNN_SIGNED_LIF_MLP_READOUT",
                "lr": 0.003,
                "lr2": 0.0001,
                "flags": {**signed_base, **flags},
            }
        )

    lif_plif_grid = [
        ("lif_plif_v05_scale25_lr3e3", {"lif_v_threshold": 0.5, "lif_input_scale": 2.5}),
        ("lif_plif_v03_scale25_lr3e3", {"lif_v_threshold": 0.3, "lif_input_scale": 2.5}),
        ("lif_plif_v03_scale40_lr3e3", {"lif_v_threshold": 0.3, "lif_input_scale": 4.0}),
        ("lif_plif_v02_scale40_lr3e3", {"lif_v_threshold": 0.2, "lif_input_scale": 4.0}),
        ("lif_plif_v02_scale60_lr3e3", {"lif_v_threshold": 0.2, "lif_input_scale": 6.0}),
        ("lif_plif_v01_scale60_lr5e3", {"lif_v_threshold": 0.1, "lif_input_scale": 6.0}),
    ]
    for name, flags in lif_plif_grid:
        lr = 0.005 if name.endswith("lr5e3") else 0.003
        specs.append(
            {
                "name": name,
                "model": "CUPY_SNN_LIF_PLIF_LIF_READOUT",
                "lr": lr,
                "lr2": 0.0001,
                "flags": {**common_best, **soft_aug, **flags},
            }
        )

    return specs


def main():
    parser = argparse.ArgumentParser(description="Long SNN experiment batch")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr2", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--stage2_select", default="train_loss", choices=["train_loss", "val_acc"])
    parser.add_argument("--stage2_patience", type=int, default=150)
    parser.add_argument("--max_trial_num", type=int, default=8)
    parser.add_argument("--max_run_name", default="max_lif_readout_seed23_adamw_lr3e3")
    parser.add_argument("--max_epoch", type=int, default=1500)
    parser.add_argument("--max_epoch2", type=int, default=600)
    parser.add_argument("--max_patience", type=int, default=200)
    parser.add_argument("--tune_trial_num", type=int, default=1)
    parser.add_argument("--tune_epoch", type=int, default=300)
    parser.add_argument("--tune_epoch2", type=int, default=100)
    parser.add_argument("--skip_tune", action="store_true")
    parser.add_argument("--log_dir", default=os.path.join("logs", "training", "big_snn_experiments"))
    parser.add_argument("--out_json", default=os.path.join("benchmarks", "results", "grid", "big_snn_experiments_seed23.json"))
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {"params": vars(args), "results": []}
    specs = build_specs(args)
    for spec in specs:
        result = run_one(args, spec)
        payload["results"].append(result)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("DONE", result["name"], result["return_code"], result["summary"].get("result_mean"), flush=True)
        if result["return_code"] != 0:
            print("Stopping after failed experiment:", result["log_path"], flush=True)
            break


if __name__ == "__main__":
    main()
