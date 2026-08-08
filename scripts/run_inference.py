"""
Run inference on sampled data with MLLMs.

Supports:
  - OpenAI GPT-4V / GPT-4o API
  - Google Gemini Vision API
  - Local HuggingFace VLM (Qwen2.5-VL, LLaVA, etc.)
  - Open-source API providers (Together, Fireworks, etc.)

Output format (JSONL):
{
    "sample_id": "pope_0001",
    "task": "pope",
    "image_path": "...",
    "question": "...",
    "ground_truth": "no",
    "model_name": "gpt-4o",
    "prompt_mode": "cot",
    "answer": "Yes",
    "rationale": "I can see a dog near the person.",
    "raw_response": "...",
    "timestamp": "2026-..."
}
"""

import json
import time
import argparse
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import yaml

from config_loader import load_configs


def load_samples(sampled_path: Path) -> list[dict]:
    """Load sampled JSONL data."""
    samples = []
    with open(sampled_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def encode_image(image_path: Path) -> str:
    """Read image file and encode as base64."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime_type


# === Model Clients ===

class OpenAIVisionClient:
    """Client for OpenAI GPT-4V / GPT-4o."""

    def __init__(self, cfg: dict):
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 256)
        self.api_key = cfg.get("api_key") or __import__("os").environ.get("OPENAI_API_KEY")
        self.api_base = cfg.get("api_base") or None

    def generate(self, image_path: Path, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        b64_img, mime = encode_image(image_path)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}},
                ],
            }],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class GeminiVisionClient:
    """Client for Google Gemini Vision API."""

    def __init__(self, cfg: dict):
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 256)
        self.api_key = cfg.get("api_key") or __import__("os").environ.get("GEMINI_API_KEY")

    def generate(self, image_path: Path, prompt: str) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        with open(image_path, "rb") as f:
            image_data = f.read()

        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_data}],
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        return response.text


class HuggingFaceVLMClient:
    """Client for local HuggingFace VLM models."""

    def __init__(self, cfg: dict):
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_new_tokens = cfg.get("max_new_tokens", 256)
        self.device = cfg.get("device", "cuda")
        self.dtype = cfg.get("dtype", "bfloat16")
        self.load_in_4bit = cfg.get("load_in_4bit", False)

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

        print(f"Loading model: {self.model_name}...")

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": getattr(torch, self.dtype),
            "trust_remote_code": True,
        }

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.dtype),
            )

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, **model_kwargs)
        print(f"Model loaded on {self.model.device}")

    def generate(self, image_path: Path, prompt: str) -> str:
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        # Qwen2.5-VL style
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]
        return response


class TogetherAPIClient:
    """Client for Together AI / Fireworks / other API providers for open-source models."""

    def __init__(self, cfg: dict):
        self.model_name = cfg["model_name"]
        self.api_key = cfg.get("api_key") or __import__("os").environ.get("TOGETHER_API_KEY")
        self.api_base = cfg.get("api_base", "https://api.together.xyz/v1")
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 256)

    def generate(self, image_path: Path, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        b64_img, mime = encode_image(image_path)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}},
                ],
            }],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


def create_client(models_cfg: dict, model_type: str):
    """Factory method to create the appropriate client."""
    if model_type == "closed_source":
        cfg = models_cfg["closed_source"]
    elif model_type == "open_source":
        cfg = models_cfg["open_source"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    provider = cfg.get("provider", "").lower()

    if model_type == "closed_source":
        if provider == "openai":
            return OpenAIVisionClient(cfg), cfg["model_name"]
        elif provider == "gemini":
            return GeminiVisionClient(cfg), cfg["model_name"]
        else:
            raise ValueError(f"Unknown closed-source provider: {provider}")

    elif model_type == "open_source":
        if cfg.get("use_api", False):
            return TogetherAPIClient(cfg), cfg["model_name"]
        else:
            return HuggingFaceVLMClient(cfg), cfg["model_name"]


def normalize_pope_answer(answer: str, eval_cfg: dict) -> str:
    """Normalize yes/no answers for POPE."""
    answer_lower = answer.strip().lower()
    norm = eval_cfg.get("answer_normalization", {}).get("pope", {})

    yes_patterns = norm.get("yes_patterns", ["yes"])
    no_patterns = norm.get("no_patterns", ["no"])

    for pat in yes_patterns:
        if pat in answer_lower:
            return "yes"
    for pat in no_patterns:
        if pat in answer_lower:
            return "no"
    return "invalid"


def parse_rationale_and_answer(raw_response: str, task: str, prompt_mode: str) -> tuple[str, str]:
    """
    Extract rationale and final answer from raw response.
    For CoT: separate reasoning from final answer.
    For Direct: entire response is the answer.
    """
    if prompt_mode == "direct":
        return "", raw_response.strip()

    # For CoT, try to extract final answer
    # Look for patterns like "Final answer: yes" or "Answer: 42"
    import re

    final_answer_patterns = [
        r"(?i)final\s*answer\s*:\s*(.+)$",
        r"(?i)answer\s*:\s*(.+)$",
        r"(?i)therefore,?\s*(.+)$",
        r"(?i)so\s+the\s+answer\s+is\s*(.+)$",
    ]

    lines = raw_response.strip().split("\n")
    final_answer = lines[-1]  # Default: last line
    rationale = "\n".join(lines[:-1])

    for pat in final_answer_patterns:
        for line in lines:
            match = re.search(pat, line)
            if match:
                final_answer = match.group(1).strip()
                # Everything before this line is rationale
                idx = lines.index(line)
                rationale = "\n".join(lines[:idx])
                return rationale.strip(), final_answer

    return rationale.strip(), final_answer.strip()


def run_inference(
    client,
    model_name: str,
    samples: list[dict],
    task: str,
    prompt_mode: str,
    prompts_cfg: dict,
    eval_cfg: dict,
    image_dir: Path,
    output_path: Path,
    save_every: int = 50,
):
    """Run inference on all samples and save results."""
    results = []
    prompt_key = f"{task}_{prompt_mode}"
    prompt_template = prompts_cfg.get(prompt_key, "")

    if not prompt_template:
        raise ValueError(f"No prompt template found for key: {prompt_key}")

    for i, sample in enumerate(samples):
        # Build question and prompt
        question = sample.get("question", sample.get("text", ""))
        prompt = prompt_template.format(question=question)

        # Resolve image path
        image_path_raw = sample.get("image_path", "")
        if image_path_raw:
            image_path = Path(image_path_raw)
            if not image_path.is_absolute():
                image_path = image_dir / image_path
            if not image_path.exists():
                # Try relative to current working directory
                alt_path = Path.cwd() / image_path_raw
                if alt_path.exists():
                    image_path = alt_path
        else:
            image_path = None

        print(f"[{i+1}/{len(samples)}] Processing {task} sample {sample.get('sample_id', i)}...")

        # Retry up to 3 times with backoff
        for attempt in range(3):
            try:
                raw_response = client.generate(image_path if image_path else None, prompt)
                rationale, answer = parse_rationale_and_answer(raw_response, task, prompt_mode)
                break
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"  RETRY [{attempt+1}/3] after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"  ERROR: {e}")
                    raw_response = f"ERROR: {e}"
                    rationale, answer = "", ""

        result = {
            "sample_id": sample.get("sample_id", sample.get("pid", f"{task}_{i:04d}")),
            "task": task,
            "image_path": str(image_path),
            "question": question,
            "ground_truth": sample.get("answer", sample.get("ground_truth", "")),
            "model_name": model_name,
            "prompt_mode": prompt_mode,
            "answer": answer,
            "rationale": rationale,
            "raw_response": raw_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)

        # Save checkpoint
        if (i + 1) % save_every == 0 or i == len(samples) - 1:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        time.sleep(0.5)  # Rate limiting

    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference on sampled data")
    parser.add_argument("--task", required=True, choices=["pope", "mathvista", "logicocr"])
    parser.add_argument("--model_type", required=True, choices=["closed_source", "open_source"])
    parser.add_argument("--prompt_mode", required=True, choices=["direct", "cot"])
    parser.add_argument("--input", type=Path, required=True, help="Path to sampled JSONL")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/model_answers"))
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N samples")
    args = parser.parse_args()

    models_cfg, prompts_cfg, eval_cfg = load_configs()
    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    client, model_name = create_client(models_cfg, args.model_type)
    print(f"Using model: {model_name}")

    output_name = f"{args.task}_{model_name.replace('/', '_')}_{args.prompt_mode}.jsonl"
    output_path = args.output_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_inference(
        client=client,
        model_name=model_name,
        samples=samples,
        task=args.task,
        prompt_mode=args.prompt_mode,
        prompts_cfg=prompts_cfg,
        eval_cfg=eval_cfg,
        image_dir=args.image_dir,
        output_path=output_path,
        save_every=args.save_every,
    )

    print(f"\nDone! Results saved to {output_path}")


if __name__ == "__main__":
    main()
