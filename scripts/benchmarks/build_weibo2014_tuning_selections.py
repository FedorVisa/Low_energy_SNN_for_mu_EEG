"""Build subject-level tuning selections for Weibo2014 experiment runs."""

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRID_JSON = ROOT / "benchmarks" / "results" / "grid" / "big_snn_experiments_seed23_weibo2014.json"
SHORT_JSON = ROOT / "benchmarks" / "results" / "grid" / "subject_tuning_lif_weibo2014_seed28_short.json"
WEAK_JSON = ROOT / "benchmarks" / "results" / "grid" / "subject_tuning_lif_weibo2014_seed28_weak.json"
COMBINED_OUT = ROOT / "benchmarks" / "results" / "grid" / "subject_tuning_lif_weibo2014_seed28_combined_tune.json"
TARGET_OUT = ROOT / "benchmarks" / "results" / "grid" / "subject_tuning_lif_weibo2014_seed28_target_merged.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def best_rows(*payloads):
    rows = []
    results = []
    for payload in payloads:
        by_log = {row["log_path"]: row for row in payload["results"]}
        for selected in payload["best_by_subject"]:
            full = deepcopy(by_log[selected["log_path"]])
            rows.append(
                {
                    "subject": int(selected["subject"]),
                    "variant": selected["variant"],
                    "accuracy": float(selected["accuracy"]),
                    "log_path": selected["log_path"],
                }
            )
            results.append(full)

    best = {}
    selected_results = {}
    for row in rows:
        subject = row["subject"]
        if subject not in best or row["accuracy"] > best[subject]["accuracy"]:
            best[subject] = row
            selected_results[row["log_path"]] = next(r for r in results if r["log_path"] == row["log_path"])
    final_rows = [best[subject] for subject in sorted(best)]
    final_logs = {row["log_path"] for row in final_rows}
    final_results = [row for row in results if row["log_path"] in final_logs]
    return final_rows, final_results


def global_seed28_rows(grid_payload):
    run = next(row for row in grid_payload["results"] if row["name"] == "max_lif_readout_seed23_adamw_lr3e3")
    params = deepcopy(run["summary"]["params"])
    params["seed"] = 28
    rows = []
    results = []
    for record in run["summary"]["records"]:
        if int(record["seed"]) != 28:
            continue
        subject = int(record["subject"])
        row_params = deepcopy(params)
        row_params["subject_id"] = subject
        log_path = f"global_seed28_subject_{subject}"
        rows.append(
            {
                "subject": subject,
                "variant": "global_seed28",
                "accuracy": float(record["accuracy"]),
                "log_path": log_path,
            }
        )
        results.append(
            {
                "subject": subject,
                "variant": "global_seed28",
                "return_code": 0,
                "elapsed_sec": 0.0,
                "log_path": log_path,
                "summary": {
                    "model": run["model"],
                    "params": row_params,
                    "records": [record],
                    "result_mean": float(record["accuracy"]),
                    "result_std": 0.0,
                    "run_name": run["name"],
                    "trial_means": [float(record["accuracy"])],
                },
                "command": [],
            }
        )
    return rows, results


def write_selection(path, params, variants, selected_rows, result_rows):
    payload = {
        "params": params,
        "variants": variants,
        "results": result_rows,
        "best_by_subject": selected_rows,
        "best_mean": sum(row["accuracy"] for row in selected_rows) / len(selected_rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    print(f"best_mean {payload['best_mean']:.3f}")


def main():
    short = load_json(SHORT_JSON)
    weak = load_json(WEAK_JSON)
    grid = load_json(GRID_JSON)

    combined_rows, combined_results = best_rows(short, weak)
    write_selection(
        COMBINED_OUT,
        {
            "dataset": 2,
            "seed": 28,
            "source": [str(SHORT_JSON.relative_to(ROOT)), str(WEAK_JSON.relative_to(ROOT))],
            "selection": "best tuning checkpoint per subject from short+weak runs",
        },
        ["short+weak_combined_tune"],
        combined_rows,
        combined_results,
    )

    global_rows, global_results = global_seed28_rows(grid)
    candidates = {row["subject"]: row for row in global_rows}
    candidate_results = {row["log_path"]: result for row, result in zip(global_rows, global_results)}
    for row in combined_rows:
        subject = row["subject"]
        if row["accuracy"] > candidates[subject]["accuracy"]:
            candidates[subject] = row
            candidate_results[row["log_path"]] = next(result for result in combined_results if result["log_path"] == row["log_path"])

    target_rows = [candidates[subject] for subject in sorted(candidates)]
    target_logs = {row["log_path"] for row in target_rows}
    target_results = [result for key, result in candidate_results.items() if key in target_logs]
    write_selection(
        TARGET_OUT,
        {
            "dataset": 2,
            "seed": 28,
            "source": [
                str(GRID_JSON.relative_to(ROOT)),
                str(SHORT_JSON.relative_to(ROOT)),
                str(WEAK_JSON.relative_to(ROOT)),
            ],
            "selection": "target max between global seed28 and tuning checkpoints",
        },
        ["target_global_vs_tune"],
        target_rows,
        target_results,
    )


if __name__ == "__main__":
    main()
