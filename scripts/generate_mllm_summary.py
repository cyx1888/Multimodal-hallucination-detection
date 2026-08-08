"""
Generate the full image-aware MLLM judge summary.

This script aggregates task accuracy and image-aware MLLM judge metrics
across all tasks, models, and prompt modes.

Important: task accuracy is loaded from outputs/metrics instead of being
recomputed here. The canonical evaluation logic lives in scripts/evaluate.py,
and ad hoc matching would change POPE/MathVista accuracy numbers.

Usage:
  python scripts/generate_mllm_summary.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path


TASK_ORDER = {"pope": 0, "mathvista": 1, "logicocr": 2}
MODEL_ORDER = {"gpt-5.4": 0, "Qwen_Qwen3-VL-8B-Instruct": 1}
PROMPT_ORDER = {"direct": 0, "cot": 1}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_key(filename):
    """Extract (task, model, prompt) from a judged JSONL filename."""
    name = filename.stem
    if name.endswith("_mllm_judged"):
        name = name[: -len("_mllm_judged")]
    elif name.endswith("_judged"):
        name = name[: -len("_judged")]

    parts = name.split("_")
    task = parts[0]
    prompt = parts[-1]
    model = "_".join(parts[1:-1])
    return task, model, prompt


def sort_key(item):
    task, model, prompt = item[0]
    return (
        TASK_ORDER.get(task, 99),
        MODEL_ORDER.get(model, 99),
        PROMPT_ORDER.get(prompt, 99),
    )


def pct(value):
    if value is None or value == "":
        return "N/A"
    return f"{value * 100:.2f}%"


def short_model(model):
    return "Qwen3-VL-8B" if "Qwen" in model else model


def format_type_dist(type_dist):
    if not type_dist:
        return "-"
    return ", ".join(f"{k}:{v}" for k, v in sorted(type_dist.items()))


def load_task_metrics(metrics_dir, task, model, prompt):
    """Load canonical task accuracy and POPE rule metrics if present."""
    path = metrics_dir / f"{task}_{model}_{prompt}_metrics.json"
    if not path.exists():
        return {
            "accuracy": None,
            "correct": None,
            "total": None,
            "rule_metrics": None,
            "source": "missing",
        }

    metrics = json.loads(path.read_text(encoding="utf-8"))
    perf = metrics.get("task_performance", {})
    correct = perf.get("correct")
    if correct is None and "confusion_matrix" in perf:
        cm = perf["confusion_matrix"]
        correct = cm.get("TP", 0) + cm.get("TN", 0)
    return {
        "accuracy": perf.get("accuracy"),
        "correct": correct,
        "total": perf.get("total", perf.get("total_valid")),
        "rule_metrics": metrics.get("rule_hallucination"),
        "source": str(path),
    }


def compute_mllm_metrics(rows):
    total = len(rows)
    hall = sum(1 for r in rows if r.get("mllm_judge_hallucination") is True)
    nonhall = sum(1 for r in rows if r.get("mllm_judge_hallucination") is False)
    unknown = total - hall - nonhall

    type_dist = defaultdict(int)
    for r in rows:
        if r.get("mllm_judge_hallucination") is True:
            type_dist[r.get("mllm_judge_hallucination_type", "unknown")] += 1

    return {
        "total": total,
        "hallucinated": hall,
        "non_hallucinated": nonhall,
        "unknown": unknown,
        "hallucination_rate": round(hall / total, 4) if total else None,
        "type_distribution": dict(type_dist),
        "type_percentages": {
            k: round(v / hall * 100, 1) for k, v in type_dist.items()
        } if hall else {},
    }


def main():
    base = Path(".")
    mllm_dir = base / "outputs" / "mllm_judge_full"
    metrics_dir = base / "outputs" / "metrics"

    results = {}
    for path in sorted(mllm_dir.glob("*_mllm_judged.jsonl")):
        if path.name.startswith("smoke"):
            continue

        task, model, prompt = extract_key(path)
        rows = load_jsonl(path)
        canonical = load_task_metrics(metrics_dir, task, model, prompt)
        mllm_metrics = compute_mllm_metrics(rows)

        key = (task, model, prompt)
        results[key] = {
            "n_samples": len(rows),
            "accuracy": {
                "total": canonical["total"],
                "correct": canonical["correct"],
                "accuracy": canonical["accuracy"],
                "source": canonical["source"],
            },
            "mllm_judge": mllm_metrics,
        }
        if canonical["rule_metrics"] is not None:
            results[key]["rule_detector"] = canonical["rule_metrics"]

    csv_path = mllm_dir / "full_mllm_judge_summary.csv"
    json_path = mllm_dir / "full_mllm_judge_summary.json"

    csv_rows = []
    for key, data in sorted(results.items(), key=sort_key):
        task, model, prompt = key
        mllm = data.get("mllm_judge") or {}
        acc = data.get("accuracy") or {}
        mllm_hr = mllm.get("hallucination_rate")

        csv_rows.append({
            "Task": task,
            "Model": short_model(model),
            "Prompt": prompt,
            "N": data.get("n_samples"),
            "Accuracy": acc.get("accuracy"),
            "Correct": acc.get("correct"),
            "MLLM_HR": mllm_hr,
            "MLLM_Hall": mllm.get("hallucinated"),
            "MLLM_Unknown": mllm.get("unknown"),
            "MLLM_Type_Dist": format_type_dist(mllm.get("type_distribution", {})),
        })

    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
    except PermissionError:
        csv_path = mllm_dir / "full_mllm_judge_summary_corrected.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    json_output = {}
    for key, data in sorted(results.items(), key=sort_key):
        task, model, prompt = key
        json_output.setdefault(task, {})
        json_output[task][f"{model}/{prompt}"] = data

    try:
        json_path.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")
    except PermissionError:
        json_path = mllm_dir / "full_mllm_judge_summary_corrected.json"
        json_path.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"CSV saved: {csv_path}")
    print(f"JSON saved: {json_path}")
    print()
    print("| Task | Model | Prompt | N | Acc | Image-aware MLLM HR | MLLM Types |")
    print("|---|---|---|---:|---:|---:|---|")
    for row in csv_rows:
        print(
            f"| {row['Task']} | {row['Model']} | {row['Prompt']} | {row['N']} | "
            f"{pct(row['Accuracy'])} | {pct(row['MLLM_HR'])} | {row['MLLM_Type_Dist']} |"
        )

    print()
    print("| Task | Image-aware MLLM HR Avg. | Avg Acc |")
    print("|---|---:|---:|")
    for task in ["pope", "mathvista", "logicocr"]:
        task_rows = [r for r in csv_rows if r["Task"] == task]
        mllm_avg = sum(r["MLLM_HR"] for r in task_rows) / len(task_rows)
        acc_avg = sum(r["Accuracy"] for r in task_rows) / len(task_rows)
        print(f"| {task} | {pct(mllm_avg)} | {pct(acc_avg)} |")


if __name__ == "__main__":
    main()
