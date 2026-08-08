"""
Sample 40 items for human annotation (stratified).

Stratification:
  - POPE: 20 items
  - MathVista: 20 items
  - Both models represented
  - Direct + CoT represented
  - Hallucination + non-hallucination (by judge) both included
  - Various hallucination types if possible

Usage:
  python sample_for_annotation.py --input_dir outputs/judge_results --n 40
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_all_judged(input_dir: Path) -> list[dict]:
    """Load all judged JSONL files."""
    all_samples = []
    for f in sorted(input_dir.glob("*_judged.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    all_samples.append(json.loads(line))
    return all_samples


def sample_stratified(samples: list[dict], n: int = 40, seed: int = 42) -> list[dict]:
    """Stratified sampling for human annotation."""
    random.seed(seed)

    pope = [s for s in samples if s.get("task") == "pope"]
    mathvista = [s for s in samples if s.get("task") == "mathvista"]

    n_pope = min(20, len(pope))
    n_mv = min(20, len(mathvista))
    # Adjust if not enough samples
    if len(pope) < 20:
        n_mv = n - len(pope)
    if len(mathvista) < 20:
        n_pope = n - len(mathvista)

    selected = []
    selected += _sample_task_stratified(pope, n_pope, seed)
    selected += _sample_task_stratified(mathvista, n_mv, seed + 1)

    random.shuffle(selected)
    return selected


def _sample_task_stratified(samples: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify within a task to get approximately n samples."""
    random.seed(seed)

    # Group by model
    by_model = defaultdict(list)
    for s in samples:
        model = s.get("model_name", "unknown")
        by_model[model].append(s)

    n_models = len(by_model)
    selected = []

    for model, items in by_model.items():
        # Within model, stratify by prompt_mode and hallucination
        by_prompt = defaultdict(list)
        for s in items:
            pm = s.get("prompt_mode", "unknown")
            by_prompt[pm].append(s)

        n_prompts = len(by_prompt)
        # Target per (model, prompt, hall_category) cell
        target_per_cell = max(4, n // n_models // n_prompts // 2)

        for pm, pm_items in by_prompt.items():
            hall = [s for s in pm_items if s.get("judge_hallucination") is True]
            nonhall = [s for s in pm_items if s.get("judge_hallucination") is False]

            if hall:
                selected.extend(random.sample(hall, min(target_per_cell, len(hall))))
            if nonhall:
                selected.extend(random.sample(nonhall, min(target_per_cell, len(nonhall))))

    random.shuffle(selected)
    # Ensure we don't exceed n
    return selected[:n]


def build_annotation_template(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build blind annotation records and a separate key file.

    Returns:
      (annotation_records, key_records)
      annotation_records: clean records for human annotation (no model/judge info)
      key_records: mapping file with hidden fields for later reveal
    """
    annotation_records = []
    key_records = []
    for i, s in enumerate(samples):
        annot_id = f"annot_{i+1:03d}"

        # Blind annotation record (visible to annotator)
        record = {
            "annotation_id": annot_id,
            "sample_id": s.get("sample_id", ""),
            "task": s.get("task", ""),
            "image_path": s.get("image_path", ""),
            "question": s.get("question", ""),
            "ground_truth": s.get("ground_truth", ""),
            "model_answer": s.get("answer", ""),
            "model_rationale": s.get("rationale", "")[:500],
            # Human label placeholders
            "human_is_hallucination": None,
            "human_hallucination_type": "",
            "human_confidence": 0.0,
            "human_notes": "",
        }
        annotation_records.append(record)

        # Key record (hidden, to merge after annotation)
        key_record = {
            "annotation_id": annot_id,
            "model_name": s.get("model_name", ""),
            "prompt_mode": s.get("prompt_mode", ""),
            "judge_hallucination": s.get("judge_hallucination"),
            "judge_hallucination_type": s.get("judge_hallucination_type", ""),
            "judge_confidence": s.get("judge_confidence", 0.0),
            "judge_reason": s.get("judge_reason", ""),
        }
        key_records.append(key_record)

    return annotation_records, key_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("outputs/judge_results"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/human_annotation.jsonl"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_samples = load_all_judged(args.input_dir)
    print(f"Loaded {len(all_samples)} judged samples")

    sampled = sample_stratified(all_samples, n=args.n, seed=args.seed)
    print(f"Sampled {len(sampled)} for annotation")

    # Show sampling stats
    stats = defaultdict(lambda: defaultdict(int))
    for s in sampled:
        stats["task"][s.get("task", "?")] += 1
        stats["model"][s.get("model_name", "?")] += 1
        stats["prompt"][s.get("prompt_mode", "?")] += 1
        hall = "hall" if s.get("judge_hallucination") else "nonhall"
        stats["judge_label"][hall] += 1

    print("Sampling distribution:")
    for category, counts in stats.items():
        print(f"  {category}: {dict(counts)}")

    # Build annotation template (separate blind + key files)
    annot_records, key_records = build_annotation_template(sampled)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save blind annotation file
    with open(args.output, "w", encoding="utf-8") as f:
        for r in annot_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save key mapping file (hidden fields)
    key_path = args.output.parent / "human_annotation_key.jsonl"
    with open(key_path, "w", encoding="utf-8") as f:
        for r in key_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Annotation template saved to {args.output}")
    print(f"Annotation key file saved to {key_path}")
    print(f"\nIMPORTANT: {args.output.name} is the BLIND file for annotation.")
    print(f"          {key_path.name} contains hidden fields (model, judge).")
    print(f"          After annotation, merge by annotation_id.")
    print("\nAnnotation fields to fill:")
    print("  human_is_hallucination: true/false")
    print("  human_hallucination_type: visual_grounding_error / factual_inconsistency / reasoning_hallucination / none")
    print("  human_confidence: 0.0-1.0")
    print("  human_notes: free text")


if __name__ == "__main__":
    main()
