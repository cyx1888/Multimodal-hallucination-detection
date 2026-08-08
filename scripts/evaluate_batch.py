"""Batch evaluate all judged files with robust normalization."""
import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter
from config_loader import load_configs

sys.stdout.reconfigure(encoding='utf-8')

# Threshold: if judge_unknown_rate exceeds this, mark judge results as N/A
JUDGE_UNKNOWN_THRESHOLD = 0.20


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def normalize_pope_answer(answer: str) -> str:
    """Robust normalization of POPE answers to yes/no/invalid."""
    a = answer.strip().lower()

    # Direct yes/no variants
    if a in ("yes", "no"):
        return a

    # Handle period: "Yes." "No." "yes." "no."
    if a in ("yes.", "no."):
        return a[:-1]

    # Handle capitalization variants
    if a == "yes":
        return "yes"
    if a == "no":
        return "no"

    # Handle "Final answer: yes/no" patterns
    import re
    m = re.search(r'(?:final\s*answer|answer)\s*[:=]\s*(yes|no)\b', a)
    if m:
        return m.group(1)

    # Handle "The answer is yes/no"
    m = re.search(r'\b(yes|no)\b', a)
    if m:
        word = m.group(1)
        # Only return if yes/no is the last meaningful word or dominates
        if a.endswith(word) or a.endswith(word + '.'):
            return word

    # Substring check (last resort)
    if "yes" in a and "no" not in a:
        return "yes"
    if "no" in a and "yes" not in a:
        return "no"

    return "invalid"


def compute_pope_metrics(samples):
    """Compute POPE metrics with robust normalization."""
    tp = fp = tn = fn = invalid = 0
    for s in samples:
        gt = str(s.get("ground_truth", "")).strip().lower()
        raw = str(s.get("answer", "")).strip()
        pred = normalize_pope_answer(raw)

        if pred == "invalid":
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
    neg = fp + tn
    n_all = len(samples)

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    obj_hall_rate = fp / neg if neg > 0 else 0
    overall_hall_rate = fp / n_all if n_all > 0 else 0

    return {
        "n_total": n_all,
        "n_valid": total,
        "n_invalid": invalid,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "object_hallucination_rate": round(obj_hall_rate, 4),
        "overall_hallucination_rate": round(overall_hall_rate, 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def compute_mathvista_metrics(samples):
    """Compute MathVista accuracy (relaxed matching)."""
    correct = invalid = 0
    for s in samples:
        gt = str(s.get("ground_truth", "")).strip().lower()
        pred = str(s.get("answer", "")).strip().lower()
        if not pred:
            invalid += 1
            continue
        if gt in pred or pred in gt:
            correct += 1
    total = len(samples)
    return {
        "n_total": total,
        "n_correct": correct,
        "n_invalid": invalid,
        "accuracy": round(correct / total, 4) if total > 0 else 0,
    }


def compute_logicocr_metrics(samples):
    """Compute LogicOCR accuracy with answer-choice normalization."""

    def extract_choice(text):
        text = str(text or "").strip()
        m = re.search(r"(?:final\s*answer|answer)\s*(?::|\uff1a)?\s*([A-D])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        return m.group(1).upper() if m else ""

    correct = invalid = 0
    for s in samples:
        gt = extract_choice(s.get("ground_truth", ""))
        pred = extract_choice(s.get("answer", ""))
        if not pred:
            invalid += 1
            continue
        if gt == pred:
            correct += 1

    total = len(samples)
    return {
        "n_total": total,
        "n_correct": correct,
        "n_invalid": invalid,
        "accuracy": round(correct / total, 4) if total > 0 else 0,
    }


def infer_task(stem):
    """Infer task name from the judged filename."""
    name = stem.lower()
    if "pope" in name:
        return "pope"
    if "mathvista" in name:
        return "mathvista"
    if "logicocr" in name:
        return "logicocr"
    return "unknown"


def infer_judge_key(samples):
    """Prefer legacy judge_* fields, then image-aware mllm_judge_* fields."""
    if any("judge_hallucination" in s for s in samples):
        return "judge"
    if any("mllm_judge_hallucination" in s for s in samples):
        return "mllm_judge"
    return "judge"


def compute_judge_metrics(samples, key="judge"):
    """Compute judge metrics with unknown tracking."""
    h_field = f"{key}_hallucination"
    t_field = f"{key}_hallucination_type"
    total = len(samples)
    hallucinated = sum(1 for s in samples if s.get(h_field) is True)
    nonhall = sum(1 for s in samples if s.get(h_field) is False)
    unknown = total - hallucinated - nonhall
    valid = hallucinated + nonhall
    unknown_rate = unknown / total if total > 0 else 0

    type_dist = Counter()
    for s in samples:
        if s.get(h_field) is True:
            type_dist[s.get(t_field, "unknown")] += 1

    rate_on_all = hallucinated / total if total > 0 else 0
    rate_on_valid = hallucinated / valid if valid > 0 else 0

    result = {
        "n_total": total,
        "n_judge_valid": valid,
        "n_judge_unknown": unknown,
        "judge_unknown_rate": round(unknown_rate, 4),
        "hallucinated": hallucinated,
        "non_hallucinated": nonhall,
        "hallucination_rate_on_all": round(rate_on_all, 4),
        "hallucination_rate_on_valid": round(rate_on_valid, 4) if valid > 0 else None,
        "type_distribution": dict(type_dist),
        "type_percentages": {t: round(c / hallucinated * 100, 1) if hallucinated > 0 else 0
                             for t, c in type_dist.items()},
    }

    # Mark as N/A if unknown rate is too high
    if unknown_rate > JUDGE_UNKNOWN_THRESHOLD:
        result["judge_reliable"] = False
        result["judge_note"] = f"N/A: judge_unknown_rate={unknown_rate:.1%} exceeds threshold {JUDGE_UNKNOWN_THRESHOLD:.0%}"
    else:
        result["judge_reliable"] = True
        result["judge_note"] = ""

    return result


def format_rate(val):
    """Format a number for table display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("outputs/judge_results"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for fpath in sorted(args.input_dir.glob("*_judged.jsonl")):
        samples = load_jsonl(fpath)
        stem = fpath.stem.replace("_judged", "")
        task = infer_task(stem)
        if task == "unknown":
            print(f"Skipping file with unknown task: {fpath}")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {stem} ({task}, {len(samples)} samples)")
        print(f"{'='*60}")

        result = {"task": task, "file": str(fpath), "n_samples": len(samples)}

        if task == "pope":
            result["task_performance"] = compute_pope_metrics(samples)
            result["rule_hallucination"] = compute_judge_metrics(samples, "rule")
        elif task == "mathvista":
            result["task_performance"] = compute_mathvista_metrics(samples)
        else:
            result["task_performance"] = compute_logicocr_metrics(samples)

        judge_key = infer_judge_key(samples)
        result["judge_hallucination"] = compute_judge_metrics(samples, judge_key)
        result["judge_key"] = judge_key

        # Print summary
        tp = result["task_performance"]
        jh = result["judge_hallucination"]

        print(f"  Accuracy: {tp['accuracy']} (n_valid={tp.get('n_valid', tp.get('n_total', '?'))})")
        if task == "pope":
            print(f"  Object Hallucination Rate: {tp['object_hallucination_rate']}")
            print(f"  Overall Hallucination Rate: {tp['overall_hallucination_rate']}")
            print(f"  Invalid answers: {tp['n_invalid']}/{tp['n_total']}")
            if "rule_hallucination" in result:
                rh = result["rule_hallucination"]
                print(f"  Rule-based: hall={rh['hallucinated']}, valid={rh['n_judge_valid']}, unknown={rh['n_judge_unknown']}")
        print(f"  Judge: hall={jh['hallucinated']}, valid={jh['n_judge_valid']}, unknown={jh['n_judge_unknown']} ({jh['judge_unknown_rate']:.1%})")
        if jh["judge_note"]:
            print(f"  *** {jh['judge_note']} ***")
        if jh["type_distribution"]:
            print(f"  Judge type dist: {jh['type_distribution']}")

        all_results[stem] = result

        # Save individual
        out_path = args.output_dir / f"{stem}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Save combined
    combined = args.output_dir / "all_metrics.json"
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'File':<50} {'Acc':>8} {'ObjHall':>8} {'Inval':>6} {'JudgeHR':>8} {'Unk%':>7} {'Reliable?':>10} {'Types'}"
    print(header)
    print("-" * 100)
    for stem, r in sorted(all_results.items()):
        task = r["task"]
        tp = r["task_performance"]
        acc = tp["accuracy"]
        n_inv = tp.get("n_invalid", 0)
        if task == "pope":
            objh = format_rate(tp.get("object_hallucination_rate"))
        else:
            objh = "-"
        jh = r["judge_hallucination"]
        jr = format_rate(jh.get("hallucination_rate_on_valid"))
        unk_pct = f"{jh['judge_unknown_rate']:.0%}"
        reliable = "YES" if jh["judge_reliable"] else "NO"
        types = ", ".join(f"{k}:{v}" for k, v in jh.get("type_distribution", {}).items()) if jh["judge_reliable"] else jh.get("judge_note", "")
        print(f"{stem:<50} {acc:>8.4f} {objh:>8} {n_inv:>6} {jr:>8} {unk_pct:>7} {reliable:>10}  {types}")

    print(f"\nAll metrics saved to {args.output_dir}")

if __name__ == "__main__":
    main()
