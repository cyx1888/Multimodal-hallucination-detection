"""
Batch inference runner for POPE and MathVista.

Usage:
  python run_batch.py --task pope --prompt direct --model closed
"""

import json
import time
import base64
import argparse
from pathlib import Path
from datetime import datetime, timezone
import yaml
from openai import OpenAI


from config_loader import load_configs as load_config


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def get_client(models_cfg: dict, model_type: str) -> tuple:
    """Create OpenAI client."""
    key = "closed_source" if model_type == "closed" else "open_source"
    cfg = models_cfg[key]
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
    return client, cfg["model_name"]


def encode_image(path: Path) -> str:
    """Base64 encode image."""
    ext = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def parse_answer(raw: str, task: str, prompt_mode: str) -> tuple[str, str]:
    """Extract (rationale, answer) from raw response."""
    raw = raw.strip()
    if prompt_mode == "direct":
        return "", raw

    # CoT: look for final answer markers
    lines = raw.split("\n")
    final = lines[-1] if lines else raw
    rationale = raw

    markers = [
        "Final answer:", "final answer:", "FINAL ANSWER:",
        "Answer:", "answer:", "Therefore,", "therefore,",
    ]
    for i, line in enumerate(lines):
        for m in markers:
            if m in line:
                idx = line.index(m)
                final = line[idx + len(m):].strip()
                rationale = "\n".join(lines[:i]).strip()
                return rationale, final

    return rationale, final


def normalize_pope_answer(answer: str) -> str:
    """Normalize POPE answer to yes/no/invalid."""
    a = answer.strip().lower()
    yes_words = ["yes", "yeah", "there is", "present", "exists", "correct", "true"]
    no_words = ["no", "not", "absent", "doesn't", "does not", "incorrect", "false", "none"]

    for w in yes_words:
        if w in a:
            return "yes"
    for w in no_words:
        if w in a:
            return "no"
    return "invalid"


def run_batch(
    client: OpenAI,
    model_name: str,
    samples: list[dict],
    task: str,
    prompt_mode: str,
    prompt_template: str,
    image_dir: Path,
    output_path: Path,
    save_every: int = 50,
):
    """Run batch inference."""
    results = []
    n = len(samples)
    t0 = time.time()

    for i, sample in enumerate(samples):
        # Build prompt
        prompt = prompt_template.format(question=sample.get("question", ""))

        # Resolve image
        img_rel = sample.get("image_path", "")
        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = image_dir / img_rel
        if not img_path.exists():
            img_path = Path.cwd() / img_rel

        if not img_path.exists():
            print(f"  [{i+1}/{n}] SKIP: image not found: {img_rel}")
            continue

        # Call API with retry
        raw = None
        for attempt in range(3):
            try:
                b64, mime = encode_image(img_path)
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ]}],
                    temperature=0.0,
                    max_tokens=256,
                    timeout=30,
                )
                raw = resp.choices[0].message.content
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  [{i+1}/{n}] retry {attempt+1}: {e}")
                    time.sleep(3)
                else:
                    raw = f"ERROR: {e}"
                    print(f"  [{i+1}/{n}] FAILED: {e}")

        rationale, answer = parse_answer(raw or "", task, prompt_mode)
        if task == "pope":
            answer = normalize_pope_answer(answer)

        result = {
            "sample_id": sample.get("sample_id", f"{task}_{i:04d}"),
            "task": task,
            "image_path": str(img_path),
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "model_name": model_name,
            "prompt_mode": prompt_mode,
            "answer": answer,
            "rationale": rationale,
            "raw_response": raw,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)

        # Progress
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (n - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{n}] {sample.get('sample_id')} | ans={answer[:30]} | {rate:.1f}/s | ETA {eta:.0f}s")

        # Checkpoint
        if (i + 1) % save_every == 0 or i == n - 1:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  --- checkpoint: {len(results)} records saved ---")

        time.sleep(0.3)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["pope", "mathvista"])
    parser.add_argument("--prompt", required=True, choices=["direct", "cot"])
    parser.add_argument("--model", required=True, choices=["closed", "open"])
    parser.add_argument("--save_every", type=int, default=25)
    args = parser.parse_args()

    models_cfg, prompts_cfg = load_config()

    client, model_name = get_client(models_cfg, args.model)
    print(f"Model: {model_name}")

    # Load data
    jsonl_path = Path(f"data/sampled/{args.task}/{args.task}_sampled.jsonl")
    samples = load_jsonl(jsonl_path)
    print(f"Loaded {len(samples)} samples from {jsonl_path}")

    # Get prompt
    prompt_key = f"{args.task}_{args.prompt}"
    prompt_template = prompts_cfg.get(prompt_key, "")
    if not prompt_template:
        print(f"ERROR: prompt '{prompt_key}' not found!")
        return

    # Output path
    output_dir = Path("outputs/model_answers")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace(":", "_")
    output_path = output_dir / f"{args.task}_{safe_name}_{args.prompt}.jsonl"

    print(f"Output: {output_path}")
    print(f"Prompt: {args.prompt} ({prompt_key})")
    print(f"Save every: {args.save_every}")
    print(f"{'='*60}")

    image_dir = Path(f"data/sampled/{args.task}/images")

    run_batch(
        client=client,
        model_name=model_name,
        samples=samples,
        task=args.task,
        prompt_mode=args.prompt,
        prompt_template=prompt_template,
        image_dir=image_dir,
        output_path=output_path,
        save_every=args.save_every,
    )

    print(f"\nDONE: {args.task} {args.prompt} -> {output_path}")


if __name__ == "__main__":
    main()
