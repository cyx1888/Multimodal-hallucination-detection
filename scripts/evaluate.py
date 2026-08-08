"""
Evaluate model performance and hallucination metrics.

Computes:
  - Task performance: Accuracy, Precision, Recall, F1
  - Hallucination rates: overall, object hallucination rate (POPE)
  - Hallucination type distribution
  - Detector-human agreement (if annotation file provided)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional
import yaml


from config_loader import load_configs as _load_configs

def load_configs():
    _, _, eval_cfg = _load_configs()
    return eval_cfg


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def normalize_answer(answer: str, task: str) -> str:
    """Normalize answer for comparison."""
    return str(answer).strip().lower()


def compute_pope_metrics(samples: list[dict]) -> dict:
    """
    Compute POPE-specific metrics:
    - Accuracy, Precision, Recall, F1 (treating "yes" as positive)
    - Object Hallucination Rate: FP / (FP + TN) = FP / negative samples
    - Yes Rate
    """
    tp = fp = tn = fn = 0
    invalid = 0

    def extract_yes_no(text: str) -> Optional[str]:
        """Extract a yes/no answer from short POPE responses."""
        import re

        text = normalize_answer(str(text), "pope")
        if text == "invalid":
            return "invalid"
        m = re.search(r"\b(yes|no)\b", text)
        return m.group(1) if m else ""

    for s in samples:
        gt = normalize_answer(str(s.get("ground_truth", "")), "pope")
        pred = extract_yes_no(str(s.get("answer", "")))

        if pred == "invalid" or pred not in ("yes", "no"):
            invalid += 1
            continue

        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1

    total = tp + fp + tn + fn
    neg_samples = fp + tn  # ground truth = no

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_rate = (tp + fp) / total if total > 0 else 0
    obj_hallucination_rate = fp / neg_samples if neg_samples > 0 else 0

    return {
        "total_valid": total,
        "invalid": invalid,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "yes_rate": round(yes_rate, 4),
        "object_hallucination_rate": round(obj_hallucination_rate, 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def compute_mathvista_metrics(samples: list[dict]) -> dict:
    """Compute MathVista accuracy (relaxed matching)."""
    correct = 0
    invalid = 0
    total = len(samples)

    for s in samples:
        gt = normalize_answer(str(s.get("ground_truth", "")), "mathvista")
        pred = normalize_answer(str(s.get("answer", "")), "mathvista")

        if not pred:
            invalid += 1
            continue

        # Relaxed match: pred contains gt or gt contains pred
        if gt in pred or pred in gt:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": round(accuracy, 4),
    }


def compute_logicocr_metrics(samples: list[dict]) -> dict:
    """Compute LogicOCR accuracy (choice-based matching)."""
    import re

    def extract_choice(text: str) -> str:
        text = str(text).strip()
        # Try to find standalone A-D
        m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Try "answer: X" or "final answer: X"
        m = re.search(r"(?:answer|final answer)\s*(?::|\uff1a)\s*([A-D])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return text.lower().strip()

    correct = 0
    invalid = 0
    total = len(samples)

    for s in samples:
        gt = str(s.get("ground_truth", "")).strip()
        pred = str(s.get("answer", "")).strip()

        gt_choice = extract_choice(gt)
        pred_choice = extract_choice(pred)

        if not pred_choice or pred_choice not in ("A", "B", "C", "D"):
            invalid += 1
            continue

        if gt_choice == pred_choice:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": round(accuracy, 4),
    }


def compute_hallucination_metrics(samples: list[dict], detector_key: str = "judge") -> dict:
    """
    Compute hallucination metrics.
    detector_key: "rule" or "judge"
    """
    halluc_field = f"{detector_key}_hallucination"
    type_field = f"{detector_key}_hallucination_type"

    total = len(samples)
    hallucinated = sum(1 for s in samples if s.get(halluc_field) is True)
    non_hallucinated = sum(1 for s in samples if s.get(halluc_field) is False)
    unknown = total - hallucinated - non_hallucinated

    # Type distribution
    type_dist = Counter()
    for s in samples:
        if s.get(halluc_field) is True:
            t = s.get(type_field, "unknown")
            type_dist[t] += 1

    hallucination_rate = hallucinated / total if total > 0 else 0

    return {
        "total": total,
        "hallucinated": hallucinated,
        "non_hallucinated": non_hallucinated,
        "unknown": unknown,
        "hallucination_rate": round(hallucination_rate, 4),
        "type_distribution": dict(type_dist),
        "type_percentages": {
            t: round(c / hallucinated * 100, 1) if hallucinated > 0 else 0
            for t, c in type_dist.items()
        },
    }


def compute_agreement(samples: list[dict]) -> dict:
    """
    Compute detector-human agreement.
    Expects fields:
      - human_is_hallucination (bool)
      - judge_hallucination (bool)
    """
    tp = fp = tn = fn = 0

    for s in samples:
        human = s.get("human_is_hallucination")
        judge = s.get("judge_hallucination")

        if human is None or judge is None:
            continue

        if human and judge:
            tp += 1
        elif not human and judge:
            fp += 1
        elif not human and not judge:
            tn += 1
        elif human and not judge:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Cohen's Kappa
    po = accuracy
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total) if total > 0 else 0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    return {
        "total_pairs": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination metrics")
    parser.add_argument("--input", type=Path, required=True, help="Path to judged JSONL file")
    parser.add_argument("--task", required=True, choices=["pope", "mathvista", "logicocr"])
    parser.add_argument("--annotation", type=Path, help="Path to human annotation JSONL for agreement")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    eval_cfg = load_configs()
    samples = load_jsonl(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    results = {
        "task": args.task,
        "input_file": str(args.input),
        "n_samples": len(samples),
    }

    # Task performance
    if args.task == "pope":
        results["task_performance"] = compute_pope_metrics(samples)
    elif args.task == "logicocr":
        results["task_performance"] = compute_logicocr_metrics(samples)
    else:
        results["task_performance"] = compute_mathvista_metrics(samples)

    # Rule-based hallucination (POPE only)
    if args.task == "pope":
        results["rule_hallucination"] = compute_hallucination_metrics(samples, "rule")

    # Judge hallucination
    results["judge_hallucination"] = compute_hallucination_metrics(samples, "judge")

    # Human-judge agreement
    if args.annotation:
        annot_samples = load_jsonl(args.annotation)
        results["detector_human_agreement"] = compute_agreement(annot_samples)

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {args.task}")
    print("=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # Save
    output_name = args.input.stem.replace("_judged", "") + "_metrics.json"
    output_path = args.output_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
