"""
Analyze error patterns and generate failure case reports.

Generates:
  - CoT impact analysis: accuracy & hallucination rate with/without CoT
  - Model comparison tables
  - Hallucination type distribution by task/model/prompt
  - Failure case extraction for qualitative analysis
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
import yaml


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def group_by_experiment(samples: list[dict]) -> dict:
    """Group samples by (task, model, prompt_mode)."""
    groups = defaultdict(list)
    for s in samples:
        key = (s.get("task", "?"), s.get("model_name", "?"), s.get("prompt_mode", "?"))
        groups[key].append(s)
    return dict(groups)


def extract_logicocr_choice(text: str) -> str:
    """Extract a multiple-choice answer from LogicOCR output."""
    text = str(text or "").strip()
    m = re.search(r"(?:final answer|answer)\s*(?::|\uff1a)?\s*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


def compute_accuracy(samples: list[dict], task: str) -> float:
    """Compute task-specific accuracy for CoT/direct comparison."""
    if not samples:
        return 0

    if task == "pope":
        tp = fp = tn = fn = 0
        for s in samples:
            gt = str(s.get("ground_truth", "")).strip().lower()
            pred = str(s.get("answer", "")).strip().lower()
            if pred not in ("yes", "no"):
                if "yes" in pred:
                    pred = "yes"
                elif "no" in pred:
                    pred = "no"
            if gt == "yes" and pred == "yes":
                tp += 1
            elif gt == "no" and pred == "yes":
                fp += 1
            elif gt == "no" and pred == "no":
                tn += 1
            elif gt == "yes" and pred == "no":
                fn += 1
        total = tp + fp + tn + fn
        return (tp + tn) / total if total > 0 else 0

    if task == "logicocr":
        correct = 0
        for s in samples:
            gt = extract_logicocr_choice(s.get("ground_truth", ""))
            pred = extract_logicocr_choice(s.get("answer", ""))
            if gt and pred and gt == pred:
                correct += 1
        return correct / len(samples)

    correct = 0
    for s in samples:
        gt = str(s.get("ground_truth", "")).strip().lower()
        pred = str(s.get("answer", "")).strip().lower()
        if gt and (gt in pred or pred in gt):
            correct += 1
    return correct / len(samples)


def analyze_cot_impact(grouped: dict) -> list[dict]:
    """Compare direct vs CoT for each (task, model) pair."""
    results = []

    # Collect all (task, model) pairs
    pairs = set()
    for task, model, prompt in grouped:
        pairs.add((task, model))

    for task, model in sorted(pairs):
        direct = grouped.get((task, model, "direct"), [])
        cot = grouped.get((task, model, "cot"), [])

        direct_acc = compute_accuracy(direct, task)
        cot_acc = compute_accuracy(cot, task)
        direct_hr = sum(1 for s in direct if s.get("judge_hallucination") is True) / len(direct) if direct else 0
        cot_hr = sum(1 for s in cot if s.get("judge_hallucination") is True) / len(cot) if cot else 0

        results.append({
            "task": task,
            "model": model,
            "direct_accuracy": round(direct_acc, 4),
            "cot_accuracy": round(cot_acc, 4),
            "accuracy_delta": round(cot_acc - direct_acc, 4),
            "direct_hallucination_rate": round(direct_hr, 4),
            "cot_hallucination_rate": round(cot_hr, 4),
            "hallucination_rate_delta": round(cot_hr - direct_hr, 4),
        })

    return results


def analyze_type_distribution(grouped: dict) -> list[dict]:
    """Analyze hallucination type distribution for each experiment group."""
    results = []

    for (task, model, prompt), samples in sorted(grouped.items()):
        type_counts = Counter()
        hallucinated = 0
        for s in samples:
            if s.get("judge_hallucination") is True:
                hallucinated += 1
                t = s.get("judge_hallucination_type", "unknown")
                type_counts[t] += 1

        result = {
            "task": task,
            "model": model,
            "prompt_mode": prompt,
            "total_samples": len(samples),
            "hallucinated": hallucinated,
            "hallucination_rate": round(hallucinated / len(samples), 4) if samples else 0,
            "type_distribution": dict(type_counts),
        }
        results.append(result)

    return results


def find_failure_cases(samples: list[dict], n_per_type: int = 2) -> list[dict]:
    """
    Extract representative failure cases:
    - visual_grounding_error examples
    - reasoning_hallucination examples
    - judge-human disagreement examples
    """
    failures = {"visual_grounding_error": [], "factual_inconsistency": [], "reasoning_hallucination": [], "disagreement": []}

    for s in samples:
        t = s.get("judge_hallucination_type", "")
        if t in failures and s.get("judge_hallucination") is True:
            failures[t].append(s)

        # Judge-human disagreement
        human = s.get("human_is_hallucination")
        judge = s.get("judge_hallucination")
        if human is not None and judge is not None and human != judge:
            failures["disagreement"].append(s)

    selected = []
    for ftype, cases in failures.items():
        # Prefer cases with high confidence
        cases_sorted = sorted(cases, key=lambda s: s.get("judge_confidence", 0), reverse=True)
        for c in cases_sorted[:n_per_type]:
            c["failure_category"] = ftype
            selected.append(c)

    return selected


def merge_human_annotations(samples: list[dict], annotations: list[dict]) -> None:
    """Attach human labels to judged samples without crossing model/prompt variants."""
    exact_map = {}
    by_sample = defaultdict(list)

    for ann in annotations:
        sample_id = ann.get("sample_id")
        model = ann.get("model_name")
        prompt = ann.get("prompt_mode")
        if sample_id and model and prompt:
            exact_map[(sample_id, model, prompt)] = ann
        if sample_id:
            by_sample[sample_id].append(ann)

    unique_sample_map = {
        sample_id: rows[0]
        for sample_id, rows in by_sample.items()
        if len(rows) == 1
    }

    for s in samples:
        ann = exact_map.get((s.get("sample_id"), s.get("model_name"), s.get("prompt_mode")))
        if ann is None:
            ann = unique_sample_map.get(s.get("sample_id"))
        if ann is not None:
            s["human_is_hallucination"] = ann.get("human_is_hallucination")
            s["human_hallucination_type"] = ann.get("human_hallucination_type")


def print_cot_table(cot_results: list[dict]):
    """Print CoT impact table."""
    print("\n" + "=" * 100)
    print("CoT IMPACT ANALYSIS")
    print("=" * 100)
    header = f"{'Task':<12} {'Model':<25} {'Direct Acc':>10} {'CoT Acc':>10} {'Delta':>8} {'Direct HR':>10} {'CoT HR':>10} {'HR Delta':>8}"
    print(header)
    print("-" * 100)
    for r in cot_results:
        print(f"{r['task']:<12} {r['model']:<25} {r['direct_accuracy']:>10.4f} {r['cot_accuracy']:>10.4f} {r['accuracy_delta']:>+8.4f} {r['direct_hallucination_rate']:>10.4f} {r['cot_hallucination_rate']:>10.4f} {r['hallucination_rate_delta']:>+8.4f}")


def print_type_table(type_results: list[dict]):
    """Print type distribution table."""
    print("\n" + "=" * 100)
    print("HALLUCINATION TYPE DISTRIBUTION")
    print("=" * 100)
    for r in type_results:
        print(f"\n{ r['task']} | {r['model']} | {r['prompt_mode']}")
        print(f"  Hallucination Rate: {r['hallucination_rate']:.4f} ({r['hallucinated']}/{r['total_samples']})")
        for t, c in r['type_distribution'].items():
            print(f"    {t}: {c} ({c/r['hallucinated']*100:.1f}%)" if r['hallucinated'] > 0 else f"    {t}: {c}")


def main():
    parser = argparse.ArgumentParser(description="Analyze errors and generate reports")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing judged JSONL files")
    parser.add_argument("--annotation", type=Path, help="Human annotation JSONL for agreement analysis")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--n_failure_cases", type=int, default=3, help="Failure cases per type")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all judged files
    all_samples = []
    for f in args.input_dir.glob("*_judged.jsonl"):
        print(f"Loading {f}...")
        all_samples.extend(load_jsonl(f))

    if args.annotation:
        annot_samples = load_jsonl(args.annotation)
        merge_human_annotations(all_samples, annot_samples)

    print(f"Total samples loaded: {len(all_samples)}")
    grouped = group_by_experiment(all_samples)
    print(f"Experiment groups: {len(grouped)}")

    # Analysis 1: CoT Impact
    cot_results = analyze_cot_impact(grouped)
    print_cot_table(cot_results)
    with open(args.output_dir / "cot_impact.json", "w", encoding="utf-8") as f:
        json.dump(cot_results, f, indent=2)

    # Analysis 2: Type Distribution
    type_results = analyze_type_distribution(grouped)
    print_type_table(type_results)
    with open(args.output_dir / "type_distribution.json", "w", encoding="utf-8") as f:
        json.dump(type_results, f, indent=2)

    # Analysis 3: Failure Cases
    failure_cases = find_failure_cases(all_samples, n_per_type=args.n_failure_cases)
    print(f"\nExtracted {len(failure_cases)} failure cases")
    for fc in failure_cases:
        print(f"  [{fc['failure_category']}] {fc.get('sample_id')}: {fc.get('question', '')[:80]}")
    with open(args.output_dir / "failure_cases.json", "w", encoding="utf-8") as f:
        json.dump(failure_cases, f, indent=2, ensure_ascii=False)

    print(f"\nAll analysis results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
