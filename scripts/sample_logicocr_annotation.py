"""
Sample 20 LogicOCR items for human annotation (stratified by model/prompt/hallucination type).

Stratification plan (4 groups x 5 samples each):
  - gpt-5.4 / direct:  3 factual_inc + 2 nonhall
  - gpt-5.4 / CoT:     2 factual_inc + 1 reasoning + 2 nonhall
  - Qwen   / direct:   2 factual_inc + 1 reasoning + 2 nonhall
  - Qwen   / CoT:      2 reasoning + 2 factual_inc + 1 nonhall

Usage:
  python scripts/sample_logicocr_annotation.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict


def main():
    random.seed(42)

    judge_dir = Path("outputs/judge_results")
    all_samples = []
    for f in sorted(judge_dir.glob("logicocr_*_judged.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    all_samples.append(json.loads(line))
    print(f"Loaded {len(all_samples)} LogicOCR judged samples")

    # Group by (model, prompt_mode)
    groups = defaultdict(list)
    for s in all_samples:
        key = (s.get("model_name", "?"), s.get("prompt_mode", "?"))
        groups[key].append(s)

    # Print group stats
    for k, v in sorted(groups.items()):
        hall = sum(1 for s in v if s.get("judge_hallucination") is True)
        nonhall = sum(1 for s in v if s.get("judge_hallucination") is False)
        types = defaultdict(int)
        for s in v:
            if s.get("judge_hallucination") is True:
                types[s.get("judge_hallucination_type", "?")] += 1
        print(f"  {k[0]:30s} | {k[1]:6s} | total={len(v):3d} | hall={hall:2d} | nonhall={nonhall:2d} | types={dict(types)}")

    # Sampling plan per group: list of (hallucination_type or None=nonhall, count)
    sampling_plan = {
        ("gpt-5.4", "direct"): [
            ("factual_inconsistency", 3),
            (None, 2),
        ],
        ("gpt-5.4", "cot"): [
            ("factual_inconsistency", 2),
            ("reasoning_hallucination", 1),
            (None, 2),
        ],
        ("Qwen/Qwen3-VL-8B-Instruct", "direct"): [
            ("factual_inconsistency", 2),
            ("reasoning_hallucination", 1),
            (None, 2),
        ],
        ("Qwen/Qwen3-VL-8B-Instruct", "cot"): [
            ("reasoning_hallucination", 2),
            ("factual_inconsistency", 2),
            (None, 1),
        ],
    }

    selected = []
    for key, plan in sampling_plan.items():
        group_samples = groups[key]
        hall_by_type = defaultdict(list)
        nonhall = []
        for s in group_samples:
            if s.get("judge_hallucination") is True:
                t = s.get("judge_hallucination_type", "unknown")
                hall_by_type[t].append(s)
            else:
                nonhall.append(s)

        for hall_type, n in plan:
            if hall_type is None:
                pool = nonhall
            else:
                pool = hall_by_type.get(hall_type, [])
            picked = random.sample(pool, min(n, len(pool)))
            selected.extend(picked)

    random.shuffle(selected)
    print(f"\nSelected {len(selected)} samples for LogicOCR annotation\n")

    # Print selection distribution
    stats = defaultdict(lambda: defaultdict(int))
    for s in selected:
        model = s.get("model_name", "?")
        prompt = s.get("prompt_mode", "?")
        hall = s.get("judge_hallucination")
        htype = s.get("judge_hallucination_type", "none")
        stats[(model, prompt)]["total"] += 1
        if hall is True:
            stats[(model, prompt)][f"hall_{htype}"] += 1
        else:
            stats[(model, prompt)]["nonhall"] += 1

    header = "%-30s | %-6s | Distribution" % ("Model", "Prompt")
    print(header)
    print("-" * 70)
    for (model, prompt), dist in sorted(stats.items()):
        print("%-30s | %-6s | %s" % (model, prompt, dict(dist)))

    # Build annotation records (blind) and key records (hidden)
    annot_records = []
    key_records = []
    start_idx = 41  # Existing annotations are 001-040

    for i, s in enumerate(selected):
        annot_id = "annot_%03d" % (start_idx + i)

        record = {
            "annotation_id": annot_id,
            "sample_id": s.get("sample_id", ""),
            "task": "logicocr",
            "image_path": s.get("image_path", ""),
            "question": s.get("question", ""),
            "ground_truth": s.get("ground_truth", ""),
            "model_answer": s.get("answer", ""),
            "model_rationale": s.get("rationale", "")[:500],
            "human_is_hallucination": None,
            "human_hallucination_type": "",
            "human_confidence": 0.0,
            "human_notes": "",
        }
        annot_records.append(record)

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

    # Save
    annot_dir = Path("data/annotations")
    annot_dir.mkdir(parents=True, exist_ok=True)

    annot_path = annot_dir / "human_annotation_logicocr.jsonl"
    with open(annot_path, "w", encoding="utf-8") as f:
        for r in annot_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    key_path = annot_dir / "human_annotation_logicocr_key.jsonl"
    with open(key_path, "w", encoding="utf-8") as f:
        for r in key_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nBlind annotation file: {annot_path} ({len(annot_records)} records)")
    print(f"Key file:            {key_path} ({len(key_records)} records)")
    print(f"Annotation IDs: annot_041 to annot_{start_idx + len(selected) - 1:03d}")
    print()
    print("Fields to fill:")
    print("  human_is_hallucination: true / false")
    print("  human_hallucination_type: visual_grounding_error / factual_inconsistency / reasoning_hallucination / none")
    print("  human_confidence: 0.0 - 1.0")
    print("  human_notes: free text")
    print()
    print("IMPORTANT: Annotate the BLIND file first, then look at the key file!")

    # Print a compact table for the sampled items.
    print("\n" + "=" * 100)
    print("SAMPLED ITEMS FOR ANNOTATION")
    print("=" * 100)
    for i, s in enumerate(selected):
        is_hall = s.get("judge_hallucination")
        htype = s.get("judge_hallucination_type", "none")
        tag = "H" if is_hall else "N"
        print(f"\n  [{tag}] annot_{start_idx + i:03d} | {s.get('model_name', '?'):30s} | {s.get('prompt_mode', '?'):6s} | type={htype}")
        print(f"      Q: {s.get('question', '')[:120]}")
        print(f"      GT: {s.get('ground_truth', '')}  |  Model: {s.get('answer', '')[:60]}")


if __name__ == "__main__":
    main()
