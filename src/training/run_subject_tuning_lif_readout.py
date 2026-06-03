"""Run subject-specific tuning jobs for the LIF readout model."""

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


def variant_specs(preset="basic"):
    common_flags = {
        "optimizer": "adamw",
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "augment_train": True,
        "aug_reverse_prob": 0.0,
    }
    if preset == "weak_round2":
        return [
            {
                "name": "stage2val_lr3e3",
                "lr": 0.003,
                "epoch2": 300,
                "stage2_select": "val_acc",
                "stage2_patience": 80,
                "flags": common_flags,
            },
            {
                "name": "stage2val_lr2e3",
                "lr": 0.002,
                "epoch2": 300,
                "stage2_select": "val_acc",
                "stage2_patience": 80,
                "flags": common_flags,
            },
            {
                "name": "weighted_lr2e3",
                "lr": 0.002,
                "epoch2": 0,
                "flags": {**common_flags, "loss": "weighted_ce"},
            },
            {
                "name": "zscore_lr2e3",
                "lr": 0.002,
                "epoch2": 0,
                "flags": {**common_flags, "zscore": True},
            },
            {
                "name": "trainstat_lr2e3",
                "lr": 0.002,
                "epoch2": 0,
                "flags": {**common_flags, "train_stat_norm": True},
            },
            {
                "name": "readout_v015_scale30_lr2e3",
                "lr": 0.002,
                "epoch2": 0,
                "flags": {
                    **common_flags,
                    "readout_v_threshold": 0.15,
                    "readout_input_scale": 3.0,
                },
            },
            {
                "name": "stronger_aug_lr2e3",
                "lr": 0.002,
                "epoch2": 0,
                "flags": {
                    **common_flags,
                    "aug_copies": 2,
                    "aug_gaussian_std_min": 0.005,
                    "aug_gaussian_std_max": 0.03,
                    "aug_scale_min": 0.9,
                    "aug_scale_max": 1.1,
                    "aug_shift_max": 10,
                },
            },
        ]
    if preset == "weibo_short":
        return [
            {
                "name": "short_lr3e3_stage2",
                "lr": 0.003,
                "epoch2": 100,
                "stage2_patience": 50,
                "flags": common_flags,
            },
            {
                "name": "short_lr2e3_light_aug",
                "lr": 0.002,
                "epoch2": 0,
                "flags": {
                    **common_flags,
                    "aug_gaussian_std_min": 0.002,
                    "aug_gaussian_std_max": 0.01,
                    "aug_scale_min": 0.98,
                    "aug_scale_max": 1.02,
                    "aug_shift_max": 3,
                },
            },
        ]
    if preset == "weibo_weak":
        return [
            {
                "name": "weak_lr1e3_stage2",
                "lr": 0.001,
                "epoch2": 200,
                "stage2_patience": 80,
                "flags": common_flags,
            },
            {
                "name": "weak_lr1e3_light_aug",
                "lr": 0.001,
                "epoch2": 0,
                "flags": {
                    **common_flags,
                    "aug_gaussian_std_min": 0.002,
                    "aug_gaussian_std_max": 0.01,
                    "aug_scale_min": 0.98,
                    "aug_scale_max": 1.02,
                    "aug_shift_max": 3,
                },
            },
            {
                "name": "weak_lr2e3_trainstat",
                "lr": 0.002,
                "epoch2": 0,
                "flags": {
                    **common_flags,
                    "train_stat_norm": True,
                },
            },
        ]
    return [
        {
            "name": "stage1_lr3e3",
            "lr": 0.003,
            "epoch2": 0,
            "flags": common_flags,
        },
        {
            "name": "stage1_lr2e3",
            "lr": 0.002,
            "epoch2": 0,
            "flags": common_flags,
        },
        {
            "name": "stage1_lr1e3",
            "lr": 0.001,
            "epoch2": 0,
            "flags": common_flags,
        },
        {
            "name": "stage1_lr3e3_light_aug",
            "lr": 0.003,
            "epoch2": 0,
            "flags": {
                **common_flags,
                "aug_gaussian_std_min": 0.002,
                "aug_gaussian_std_max": 0.01,
                "aug_scale_min": 0.98,
                "aug_scale_max": 1.02,
                "aug_shift_max": 3,
            },
        },
        {
            "name": "stage1_lr3e3_no_aug",
            "lr": 0.003,
            "epoch2": 0,
            "flags": {
                "optimizer": "adamw",
                "weight_decay": 0.0001,
                "scheduler": "cosine",
            },
        },
    ]


def run_one(args, subject, spec):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_prefix}_s{subject}_{spec['name']}"
    log_path = os.path.join(args.log_dir, f"{run_name}_{stamp}.log")
    cmd = [
        args.python,
        "run_official_lif_readout.py",
        "--dataset", str(args.dataset),
        "--device", str(args.device),
        "--model", "CUPY_SNN_LIF_READOUT",
        "--subject_id", str(subject),
        "--trial_num", "1",
        "--seed", str(args.seed),
        "--epoch", str(args.epoch),
        "--epoch2", str(spec.get("epoch2", args.epoch2)),
        "--patience", str(args.patience),
        "--stage2_select", str(spec.get("stage2_select", "train_loss")),
        "--batch_size", str(args.batch_size),
        "--lr", str(spec["lr"]),
        "--lr2", str(args.lr2),
        "--run_name", run_name,
    ]
    if "stage2_patience" in spec:
        cmd.extend(["--stage2_patience", str(spec["stage2_patience"])])
    for key, value in spec.get("flags", {}).items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    started_at = time.time()
    print("RUN", subject, spec["name"], " ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    summary = parse_summary(log_path)
    return {
        "subject": subject,
        "variant": spec["name"],
        "return_code": proc.returncode,
        "elapsed_sec": round(time.time() - started_at, 2),
        "log_path": log_path,
        "summary": summary,
        "command": cmd,
    }


def best_by_subject(results):
    best = {}
    for row in results:
        if row["return_code"] != 0:
            continue
        records = row.get("summary", {}).get("records", [])
        if not records:
            continue
        accuracy = float(records[0]["accuracy"])
        subject = int(row["subject"])
        if subject not in best or accuracy > best[subject]["accuracy"]:
            best[subject] = {
                "subject": subject,
                "variant": row["variant"],
                "accuracy": accuracy,
                "log_path": row["log_path"],
            }
    return [best[s] for s in sorted(best)]


def main():
    parser = argparse.ArgumentParser(description="Subject-wise tuning for CUPY_SNN_LIF_READOUT.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--epoch", type=int, default=1500)
    parser.add_argument("--epoch2", type=int, default=0)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr2", type=float, default=0.0001)
    parser.add_argument("--run_prefix", default="subjtune_lif_seed26")
    parser.add_argument("--preset", default="basic", choices=["basic", "weak_round2", "weibo_short", "weibo_weak"])
    parser.add_argument("--log_dir", default=os.path.join("logs", "training", "subject_tuning"))
    parser.add_argument("--out_json", default=os.path.join("benchmarks", "grid", "subject_tuning_lif_seed26.json"))
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    subjects = [int(x.strip()) for x in args.subjects.split(",") if x.strip()]
    specs = variant_specs(args.preset)
    payload = {
        "params": vars(args),
        "variants": [spec["name"] for spec in specs],
        "results": [],
        "best_by_subject": [],
        "best_mean": None,
    }
    for subject in subjects:
        for spec in specs:
            result = run_one(args, subject, spec)
            payload["results"].append(result)
            payload["best_by_subject"] = best_by_subject(payload["results"])
            if payload["best_by_subject"]:
                payload["best_mean"] = sum(row["accuracy"] for row in payload["best_by_subject"]) / len(payload["best_by_subject"])
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(
                "DONE",
                subject,
                spec["name"],
                result["return_code"],
                result.get("summary", {}).get("result_mean"),
                "best_mean",
                payload["best_mean"],
                flush=True,
            )
            if result["return_code"] != 0:
                print("Stopping after failed experiment:", result["log_path"], flush=True)
                return


if __name__ == "__main__":
    main()
