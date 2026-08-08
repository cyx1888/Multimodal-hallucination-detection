"""
Load datasets from local parquet files (already downloaded).
Sample, extract images, save as JSONL + image files.

Usage:
  python prepare_data_local.py --pope_n 300 --mathvista_n 100 --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image
import io


def load_pope_parquet(download_dir: Path) -> pd.DataFrame:
    """Load all POPE parquet files into one DataFrame."""
    pope_dir = download_dir / "POPE"
    dfs = []
    for f in sorted(pope_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    print(f"Loaded POPE: {len(result)} samples from {len(dfs)} files")
    return result


def load_mathvista_parquet(download_dir: Path) -> pd.DataFrame:
    """Load MathVista testmini parquet."""
    mv_dir = download_dir / "MathVista"
    dfs = []
    for f in sorted(mv_dir.glob("testmini*.parquet")):
        df = pd.read_parquet(f)
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    print(f"Loaded MathVista testmini: {len(result)} samples from {len(dfs)} files")

    # Filter out samples without answers
    valid = result[result["answer"].notna() & (result["answer"].astype(str).str.strip() != "")]
    print(f"  With answers: {len(valid)}")
    return valid


def extract_image(image_obj) -> Image.Image:
    """Extract PIL Image from parquet image dict {'bytes': b'...'}."""
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        return Image.open(io.BytesIO(image_obj["bytes"]))
    elif isinstance(image_obj, Image.Image):
        return image_obj
    elif isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj))
    else:
        raise TypeError(f"Unexpected image type: {type(image_obj)}")


def prepare_pope(df: pd.DataFrame, output_dir: Path, n: int = 300, balanced: bool = True, seed: int = 42):
    """Sample POPE and save."""
    print(f"\n{'='*60}")
    print("Preparing POPE...")
    print(f"{'='*60}")

    pope_dir = output_dir / "pope"
    img_dir = pope_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # Separate by answer
    yes_mask = df["answer"].str.strip().str.lower() == "yes"
    no_mask = df["answer"].str.strip().str.lower() == "no"
    df_yes = df[yes_mask]
    df_no = df[no_mask]

    if balanced:
        n_each = min(n // 2, len(df_yes), len(df_no))
        sampled_yes = df_yes.sample(n=n_each, random_state=seed)
        sampled_no = df_no.sample(n=n_each, random_state=seed)
        sampled = pd.concat([sampled_yes, sampled_no]).sample(frac=1, random_state=seed)
    else:
        sampled = df.sample(n=min(n, len(df)), random_state=seed)

    print(f"Sampled: {len(sampled)} (yes: {len(sampled_yes) if balanced else '?'}, no: {len(sampled_no) if balanced else '?'})")

    records = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        sid = f"pope_{i:04d}"
        img_path = img_dir / f"{sid}.jpg"

        img = extract_image(row["image"])
        img.save(img_path)

        record = {
            "sample_id": sid,
            "task": "pope",
            "question": str(row["question"]),
            "ground_truth": str(row["answer"]).strip().lower(),
            "image_path": str(img_path),
            "image_source": str(row.get("image_source", "")),
            "category": str(row.get("category", "")),
            "question_id": str(row.get("question_id", "")),
        }
        records.append(record)

    jsonl_path = pope_dir / "pope_sampled.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {jsonl_path}")
    return records


def prepare_mathvista(df: pd.DataFrame, output_dir: Path, n: int = 100, seed: int = 42):
    """Sample MathVista and save."""
    print(f"\n{'='*60}")
    print("Preparing MathVista...")
    print(f"{'='*60}")

    mv_dir = output_dir / "mathvista"
    img_dir = mv_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # Stratified by question_type
    by_type = defaultdict(list)
    for _, row in df.iterrows():
        qtype = row.get("question_type", "unknown")
        if pd.isna(qtype):
            qtype = "unknown"
        by_type[qtype].append(row)

    print("Question types:")
    for qt, items in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {qt}: {len(items)}")

    # Allocate proportionally
    remaining = n
    sampled_rows = []
    for qt, items in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
        alloc = max(1, int(n * len(items) / len(df)))
        alloc = min(alloc, remaining, len(items))
        if alloc > 0:
            chosen = random.sample(items, alloc)
            sampled_rows.extend(chosen)
            remaining -= alloc

    random.shuffle(sampled_rows)
    sampled_rows = sampled_rows[:n]

    records = []
    for i, row in enumerate(sampled_rows):
        sid = f"mathvista_{i:04d}"
        img_path = img_dir / f"{sid}.png"

        decoded = row.get("decoded_image")
        if decoded is not None and isinstance(decoded, dict) and "bytes" in decoded:
            img = extract_image(decoded)
            img.save(img_path)

        # Parse choices if present
        choices = row.get("choices")
        try:
            if pd.isna(choices) or choices is None:
                choices = None
            elif hasattr(choices, 'tolist'):
                choices = choices.tolist()
            else:
                choices = list(choices) if not isinstance(choices, list) else choices
        except (ValueError, TypeError):
            choices = None

        metadata = row.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = {k: (str(v) if not isinstance(v, (str, int, float, bool, list, type(None))) else v) for k, v in metadata.items()}

        record = {
            "sample_id": sid,
            "task": "mathvista",
            "pid": str(row.get("pid", "")),
            "question": str(row.get("question", "")),
            "ground_truth": str(row.get("answer", "")).strip(),
            "image_path": str(img_path),
            "question_type": str(row.get("question_type", "")),
            "answer_type": str(row.get("answer_type", "")),
            "choices": choices,
            "metadata": metadata,
        }
        records.append(record)

    jsonl_path = mv_dir / "mathvista_sampled.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    type_counts = defaultdict(int)
    for r in records:
        type_counts[r["question_type"]] += 1
    print(f"Saved {len(records)} records to {jsonl_path}")
    print(f"Type distribution: {dict(type_counts)}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets from local parquet files")
    parser.add_argument("--download_dir", type=Path, default=Path("download"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/sampled"))
    parser.add_argument("--pope_n", type=int, default=300)
    parser.add_argument("--mathvista_n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # POPE
    pope_df = load_pope_parquet(args.download_dir)
    pope_records = prepare_pope(pope_df, args.output_dir, n=args.pope_n, seed=args.seed)

    # MathVista
    mv_df = load_mathvista_parquet(args.download_dir)
    mv_records = prepare_mathvista(mv_df, args.output_dir, n=args.mathvista_n, seed=args.seed)

    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"  POPE: {len(pope_records)} samples")
    print(f"  MathVista: {len(mv_records)} samples")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
