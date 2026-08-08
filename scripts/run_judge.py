"""
Run hallucination detection using an image-aware MLLM judge and rule-based detectors.

Two detection methods:
1. Rule-based detector (POPE only): GT=no, answer=yes -> object hallucination
2. Image-aware MLLM judge: GPT/Gemini-based hallucination classifier (all tasks)

Output format (JSONL, appended to model answer records):
{
    ...original fields...,
    "rule_hallucination": true/false/null,
    "rule_hallucination_type": "object_hallucination" | "omission" | null,
    "judge_hallucination": true/false,
    "judge_hallucination_type": "visual_grounding_error" | ...,
    "judge_confidence": 0.82,
    "judge_reason": "...",
    "judge_raw_response": "..."
}
"""

import json
import re
import time
import argparse
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import yaml

from config_loader import load_configs


def load_model_answers(input_path: Path) -> list[dict]:
    samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# === Rule-Based Detector (POPE) ===

def rule_based_detect(sample: dict) -> dict:
    """
    Simple rule-based detection for POPE:
    - GT=no, answer contains yes -> object_hallucination
    - GT=yes, answer contains no -> omission (not hallucination per se)
    - else -> no hallucination
    """
    gt = str(sample.get("ground_truth", "")).strip().lower()
    answer = str(sample.get("answer", "")).strip().lower()

    if "no" in gt and ("yes" in answer or "yeah" in answer):
        return {
            "rule_hallucination": True,
            "rule_hallucination_type": "object_hallucination",
        }
    elif "yes" in gt and ("no" in answer or "not" in answer):
        return {
            "rule_hallucination": False,
            "rule_hallucination_type": "omission",
        }
    else:
        return {
            "rule_hallucination": False,
            "rule_hallucination_type": None,
        }


# === LLM-as-Judge Detector ===

class OpenAIImageJudge:
    """GPT as hallucination judge (with optional image input)."""

    def __init__(self, cfg: dict):
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 512)
        self.api_key = cfg.get("api_key") or __import__("os").environ.get("OPENAI_API_KEY")
        self.api_base = cfg.get("api_base") or None

    def judge(self, image_path: Path, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        if image_path and Path(image_path).exists():
            ext = Path(image_path).suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
            mime = mime_map.get(ext, "image/jpeg")
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}},
            ]
        else:
            content = [{"type": "text", "text": prompt}]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class GeminiImageJudge:
    """Gemini as hallucination judge."""

    def __init__(self, cfg: dict):
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 512)
        self.api_key = cfg.get("api_key") or __import__("os").environ.get("GEMINI_API_KEY")

    def judge(self, image_path: Path, prompt: str) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                image_data = f.read()
            content = [prompt, {"mime_type": "image/jpeg", "data": image_data}]
        else:
            content = [prompt]

        response = model.generate_content(
            content,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        return response.text


def create_judge_client(models_cfg: dict):
    """Create image-aware judge client from config."""
    judge_cfg = models_cfg.get("judge_mllm", models_cfg.get("closed_source"))
    provider = judge_cfg.get("provider", "").lower()

    if provider == "openai":
        return OpenAIImageJudge(judge_cfg)
    elif provider == "gemini":
        return GeminiImageJudge(judge_cfg)
    else:
        raise ValueError(f"Unknown judge provider: {provider}")


def parse_judge_json(response: str) -> dict:
    """Parse JSON from judge response, handling markdown code blocks."""
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "hallucination": None,
            "type": "parse_error",
            "confidence": 0.0,
            "brief_reason": f"Failed to parse judge response: {json_str[:200]}",
        }


def run_image_judge(
    judge_client,
    samples: list[dict],
    prompts_cfg: dict,
    save_every: int = 20,
) -> list[dict]:
    """Run image-aware MLLM judge on all samples."""
    template = prompts_cfg["judge_image_zero_shot"]

    for i, sample in enumerate(samples):
        # Use manual replace instead of format() to avoid JSON curly brace conflicts
        prompt = template
        prompt = prompt.replace("{question}", str(sample.get("question", "")))
        prompt = prompt.replace("{ground_truth}", str(sample.get("ground_truth", "")))
        prompt = prompt.replace("{model_answer}", str(sample.get("answer", "")))
        prompt = prompt.replace("{rationale}", str(sample.get("rationale", "(no rationale)")))

        image_path = sample.get("image_path", "")

        print(f"[Judge {i+1}/{len(samples)}] Judging {sample.get('sample_id', i)}...")

        try:
            raw = judge_client.judge(Path(image_path) if image_path else None, prompt)
            parsed = parse_judge_json(raw)
        except Exception as e:
            print(f"  JUDGE ERROR: {e}")
            parsed = {"hallucination": None, "type": "judge_error", "confidence": 0.0, "brief_reason": str(e)}
            raw = f"ERROR: {e}"

        hallucination = parsed.get("hallucination")
        hallucination_type = parsed.get("type", "unknown")
        confidence = parsed.get("confidence", 0.0)
        reason = parsed.get("brief_reason", "")
        image_evidence = parsed.get("image_evidence", "")

        sample["mllm_judge_hallucination"] = hallucination
        sample["mllm_judge_hallucination_type"] = hallucination_type
        sample["mllm_judge_confidence"] = confidence
        sample["mllm_judge_image_evidence"] = image_evidence
        sample["mllm_judge_reason"] = reason
        sample["mllm_judge_raw_response"] = raw
        sample["mllm_judge_method"] = "image_zero_shot"

        # Backward-compatible aliases used by evaluation scripts and older outputs.
        sample["judge_hallucination"] = hallucination
        sample["judge_hallucination_type"] = hallucination_type
        sample["judge_confidence"] = confidence
        sample["judge_image_evidence"] = image_evidence
        sample["judge_reason"] = reason
        sample["judge_raw_response"] = raw
        sample["judge_method"] = "image_zero_shot"

        time.sleep(0.5)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Run hallucination detectors")
    parser.add_argument("--input", type=Path, required=True, help="Path to model answers JSONL")
    parser.add_argument("--task", required=True, choices=["pope", "mathvista", "logicocr"])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/judge_results"))
    parser.add_argument("--rule_only", action="store_true", help="Skip the image-aware judge and run rule-based detection only")
    args = parser.parse_args()

    models_cfg, prompts_cfg, _ = load_configs()
    samples = load_model_answers(args.input)
    print(f"Loaded {len(samples)} model answers from {args.input}")

    # Step 1: Rule-based detection
    if args.task == "pope":
        for sample in samples:
            rule_result = rule_based_detect(sample)
            sample.update(rule_result)
        print("Rule-based detection complete.")

    # Step 2: Image-aware MLLM judge
    if not args.rule_only:
        judge_client = create_judge_client(models_cfg)
        samples = run_image_judge(judge_client, samples, prompts_cfg)
        print("Image-aware MLLM judge detection complete.")

    # Save results
    output_name = args.input.stem + "_judged.jsonl"
    output_path = args.output_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Done! Judge results saved to {output_path}")


if __name__ == "__main__":
    main()
