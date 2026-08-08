"""
Data preparation for MLLM hallucination evaluation.
Efficient streaming approach using shuffle + buffered sampling.

Usage:
  python prepare_data.py --pope_n 300 --mathvista_n 100 --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset


def prepare_pope(output_dir: Path, n: int = 300, balanced: bool = True, seed: int = 42):
    """Load POPE via streaming, sample efficiently, save images and JSONL."""
    print(f"\n{'='*60}")
    print("Preparing POPE dataset...")
    print(f"{'='*60}")

    pope_dir = output_dir / "pope"
    img_dir = pope_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmms-lab/POPE", streaming=True)

    # Shuffle and buffer enough for balanced sampling
    # Oversample to ensure enough pos/neg after filtering
    buffer_size = min(n * 8, 9000)  # POPE has ~9000 samples
    print(f"Streaming ~{buffer_size} random samples...")

    stream = ds["test"].shuffle(seed=seed).take(buffer_size)

    pos_samples = []
    neg_samples = []

    for sample in stream:
        ans = str(sample["answer"]).strip().lower()
        if ans == "yes" and len(pos_samples) < n // 2:
            pos_samples.append(sample)
        elif ans == "no" and len(neg_samples) < n // 2:
            neg_samples.append(sample)

        if len(pos_samples) >= n // 2 and len(neg_samples) >= n // 2:
            break

    sampled = pos_samples + neg_samples
    random.seed(seed)
    random.shuffle(sampled)
    sampled = sampled[:n]

    print(f"Sampled: {len(sampled)} (yes: {len(pos_samples)}, no: {len(neg_samples)})")

    # Save
    records = []
    for i, sample in enumerate(sampled):
        sid = f"pope_{i:04d}"
        img_path = img_dir / f"{sid}.jpg"
        sample["image"].save(img_path)

        record = {
            "sample_id": sid,
            "task": "pope",
            "question": sample["question"],
            "ground_truth": str(sample["answer"]).strip().lower(),
            "image_path": str(img_path),
            "image_source": sample.get("image_source", ""),
            "category": sample.get("category", ""),
            "question_id": sample.get("question_id", ""),
        }
        records.append(record)

    jsonl_path = pope_dir / "pope_sampled.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved to {jsonl_path}")
    return records


def prepare_mathvista(output_dir: Path, n: int = 100, seed: int = 42):
    """Load MathVista testmini via streaming, sample, save."""
    print(f"\n{'='*60}")
    print("Preparing MathVista (testmini) dataset...")
    print(f"{'='*60}")

    mv_dir = output_dir / "mathvista"
    img_dir = mv_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("AI4Math/MathVista", streaming=True)

    # testmini has ~1000 samples, shuffle and take enough
    buffer_size = min(n * 5, 1000)
    print(f"Streaming ~{buffer_size} random samples...")

    stream = ds["testmini"].shuffle(seed=seed).take(buffer_size)

    # Collect with type stratification
    by_type = defaultdict(list)
    type_alloc = {}  # Will be set after first pass
    all_collected = []

    for sample in stream:
        ans = sample.get("answer")
        if ans is None or str(ans).strip() == "":
            continue

        qtype = sample.get("question_type", "unknown")
        by_type[qtype].append(sample)
        all_collected.append(sample)

    print(f"Collected {len(all_collected)} valid samples")
    print(f"Question types: {dict((k, len(v)) for k, v in by_type.items())}")

    # Distribute n samples across types proportionally, min 1 per type
    random.seed(seed)
    sampled = []
    remaining = n

    for qtype, items in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
        alloc = max(1, int(n * len(items) / len(all_collected)))
        alloc = min(alloc, remaining, len(items))
        if alloc > 0:
            sampled.extend(random.sample(items, alloc))
            remaining -= alloc

    random.shuffle(sampled)
    sampled = sampled[:n]

    # Save
    records = []
    for i, sample in enumerate(sampled):
        sid = f"mathvista_{i:04d}"
        img_path = img_dir / f"{sid}.png"

        decoded = sample.get("decoded_image")
        if decoded is not None:
            decoded.save(img_path)

        record = {
            "sample_id": sid,
            "task": "mathvista",
            "pid": sample.get("pid", ""),
            "question": sample.get("question", ""),
            "ground_truth": str(sample.get("answer", "")).strip(),
            "image_path": str(img_path),
            "question_type": sample.get("question_type", ""),
            "answer_type": sample.get("answer_type", ""),
            "choices": sample.get("choices"),
            "metadata": sample.get("metadata", {}),
        }
        records.append(record)

    jsonl_path = mv_dir / "mathvista_sampled.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    type_counts = defaultdict(int)
    for r in records:
        type_counts[r["question_type"]] += 1
    print(f"Saved {len(records)} samples to {jsonl_path}")
    print(f"Type distribution: {dict(type_counts)}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--output_dir", type=Path, default=Path("data/sampled"))
    parser.add_argument("--pope_n", type=int, default=300)
    parser.add_argument("--mathvista_n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pope_records = prepare_pope(args.output_dir, n=args.pope_n, seed=args.seed)
    mv_records = prepare_mathvista(args.output_dir, n=args.mathvista_n, seed=args.seed)

    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"  POPE: {len(pope_records)} samples")
    print(f"  MathVista: {len(mv_records)} samples")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
