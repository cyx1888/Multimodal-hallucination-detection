"""
Sample data from POPE and MathVista datasets for evaluation.

POPE datasets expected format:
  - COCO-based POPE: JSON/JSONL with image, question, answer (yes/no), category
  - Alternative: any VQA dataset with yes/no questions about object presence

MathVista datasets:
  - Testmini / full set: JSON with pid, question, answer, image, etc.
  - Source: https://huggingface.co/datasets/AI4Math/MathVista
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_pope(data_path: Path, subset: str = "popular") -> list[dict]:
    """Load POPE dataset. Supports multiple formats."""
    if not data_path.exists():
        raise FileNotFoundError(f"POPE data not found at {data_path}")

    samples = []
    if data_path.is_dir():
        # Load all JSON files in the directory
        for f in sorted(data_path.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    samples.extend(data)
                elif isinstance(data, dict):
                    samples.extend(data.values() if not isinstance(list(data.values())[0], dict) else data.values())
    else:
        with open(data_path) as f:
            data = json.load(f)
            samples = data if isinstance(data, list) else list(data.values())

    return samples


def load_mathvista(data_path: Path) -> list[dict]:
    """Load MathVista dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"MathVista data not found at {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    # MathVista is typically a list of samples
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return list(data.values())
    return []


def sample_pope(
    samples: list[dict],
    n: int = 300,
    balanced: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Sample n items from POPE. If balanced, equal pos/neg."""
    random.seed(seed)

    pos = [s for s in samples if str(s.get("answer", s.get("label", ""))).lower() == "yes"]
    neg = [s for s in samples if str(s.get("answer", s.get("label", ""))).lower() == "no"]

    if balanced:
        n_per = min(n // 2, len(pos), len(neg))
        sampled = random.sample(pos, n_per) + random.sample(neg, n_per)
    else:
        sampled = random.sample(samples, min(n, len(samples)))

    random.shuffle(sampled)
    return sampled


def sample_mathvista(
    samples: list[dict],
    n: int = 100,
    stratified: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Sample n items from MathVista. Optionally stratify by question type."""
    random.seed(seed)

    if stratified:
        # Group by category / question_type
        by_type = defaultdict(list)
        for s in samples:
            qtype = s.get("question_type", s.get("category", "unknown"))
            by_type[qtype].append(s)

        n_per_type = max(1, n // len(by_type))
        sampled = []
        for qtype, items in by_type.items():
            sampled.extend(random.sample(items, min(n_per_type, len(items))))

        if len(sampled) < n:
            remaining = [s for s in samples if s not in sampled]
            sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

        random.shuffle(sampled)
        return sampled[:n]
    else:
        return random.sample(samples, min(n, len(samples)))


def save_samples(samples: list[dict], output_path: Path):
    """Save sampled data as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample data for hallucination evaluation")
    parser.add_argument("--pope_path", type=Path, help="Path to POPE dataset")
    parser.add_argument("--mathvista_path", type=Path, help="Path to MathVista dataset")
    parser.add_argument("--pope_n", type=int, default=300, help="Number of POPE samples")
    parser.add_argument("--mathvista_n", type=int, default=100, help="Number of MathVista samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("data/sampled"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pope_path:
        print(f"Loading POPE from {args.pope_path}...")
        pope_data = load_pope(args.pope_path)
        print(f"  Loaded {len(pope_data)} total samples")
        pope_sampled = sample_pope(pope_data, n=args.pope_n, seed=args.seed)
        save_samples(pope_sampled, args.output_dir / "pope_sampled.jsonl")
        # Show stats
        pos = sum(1 for s in pope_sampled if str(s.get("answer", s.get("label", ""))).lower() == "yes")
        print(f"  Positive: {pos}, Negative: {len(pope_sampled) - pos}")

    if args.mathvista_path:
        print(f"Loading MathVista from {args.mathvista_path}...")
        mv_data = load_mathvista(args.mathvista_path)
        print(f"  Loaded {len(mv_data)} total samples")
        mv_sampled = sample_mathvista(mv_data, n=args.mathvista_n, seed=args.seed)
        save_samples(mv_sampled, args.output_dir / "mathvista_sampled.jsonl")


if __name__ == "__main__":
    main()
