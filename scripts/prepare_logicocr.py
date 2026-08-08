"""
Prepare LogicOCR sampled data for MLMM hallucination evaluation.

Loads LogicOCR_gen.json and LogicOCR_gen.zip from local download directory,
samples N records, extracts images from zip, and outputs JSONL.
"""

import json
import random
import zipfile
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare LogicOCR sampled data")
    parser.add_argument("--n", type=int, default=50, help="Number of samples (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--data_dir", type=Path, default=Path("download/LogicOCR"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/sampled/logicocr"))
    args = parser.parse_args()

    random.seed(args.seed)

    # Load JSON data
    json_path = args.data_dir / "LogicOCR_gen.json"
    with open(json_path, encoding="utf-8") as f:
        all_samples = json.load(f)
    print(f"Loaded {len(all_samples)} samples from {json_path}")

    # Random sample
    sampled = random.sample(all_samples, min(args.n, len(all_samples)))
    print(f"Sampled {len(sampled)} records (seed={args.seed})")

    # Open zip file for image extraction
    zip_path = args.data_dir / "LogicOCR_gen.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    zf = zipfile.ZipFile(zip_path)

    # Create output directories
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, sample in enumerate(sampled):
        sample_id = f"logicocr_{i:04d}"

        # Extract image from zip
        image_filename = sample["image"]  # e.g. "0.jpg"
        zip_member = f"LogicOCR_gen/{image_filename}"
        ext = Path(image_filename).suffix.lstrip(".") or "jpg"
        output_image_name = f"{sample_id}.{ext}"
        output_image_path = image_dir / output_image_name

        with zf.open(zip_member) as src:
            with open(output_image_path, "wb") as dst:
                dst.write(src.read())

        # Build question: question + choices
        question_text = sample["question"]
        choices_text = sample.get("choices", "")
        if choices_text:
            full_question = f"{question_text}\nChoices:\n{choices_text}"
        else:
            full_question = question_text

        record = {
            "sample_id": sample_id,
            "task": "logicocr",
            "question": full_question,
            "ground_truth": sample["solution"],
            "image_path": str(output_image_path),
            "metadata": {
                "source": "LogicOCR",
                "data_source": sample.get("data_source", ""),
                "background": sample.get("background", False),
                "handwritten": sample.get("handwritten", False),
                "type": sample.get("type", []),
                "context": sample.get("context", ""),
            },
        }
        records.append(record)

    zf.close()

    # Write JSONL
    output_path = args.output_dir / "logicocr_sampled.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done! {len(records)} records -> {output_path}")
    print(f"Images saved to {image_dir}")


if __name__ == "__main__":
    main()
