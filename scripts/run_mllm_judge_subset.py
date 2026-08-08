"""
Run an image-aware MLLM judge on the human-annotated subset.

It evaluates only the 60 human-labeled samples (40 main + 20 LogicOCR),
so the image-aware judge can be compared directly with human-as-judge labels.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from config_loader import load_configs
from run_judge import OpenAIImageJudge, GeminiImageJudge, parse_judge_json


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_image_judge(cfg: dict):
    provider = cfg.get("provider", "").lower()
    cfg = dict(cfg)
    cfg["text_only"] = False
    if provider == "openai":
        return OpenAIImageJudge(cfg)
    if provider == "gemini":
        return GeminiImageJudge(cfg)
    raise ValueError(f"Unknown image judge provider: {provider}")


def fill_prompt(template: str, row: dict) -> str:
    prompt = template
    prompt = prompt.replace("{question}", str(row.get("question", "")))
    prompt = prompt.replace("{ground_truth}", str(row.get("ground_truth", "")))
    prompt = prompt.replace("{model_answer}", str(row.get("model_answer", row.get("answer", ""))))
    prompt = prompt.replace("{rationale}", str(row.get("model_rationale", row.get("rationale", "(no rationale)"))))
    return prompt


def agreement(rows: list[dict], prefix: str = "mllm_judge") -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        human = row.get("human_is_hallucination")
        judge = row.get(f"{prefix}_hallucination")
        if human is None or judge is None:
            continue
        if human and judge:
            tp += 1
        elif not human and judge:
            fp += 1
        elif not human and not judge:
            tn += 1
        elif human and not judge:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    if total:
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
        kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    else:
        kappa = None

    return {
        "N": total,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "cohens_kappa": round(kappa, 4) if kappa is not None else None,
    }


def grouped_agreement(rows: list[dict]) -> dict:
    groups = {
        "overall_60": rows,
        "main_40": [r for r in rows if r.get("task") in {"pope", "mathvista"}],
        "pope_20": [r for r in rows if r.get("task") == "pope"],
        "mathvista_20": [r for r in rows if r.get("task") == "mathvista"],
        "logicocr_20": [r for r in rows if r.get("task") == "logicocr"],
    }
    return {name: agreement(group) for name, group in groups.items()}


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def print_summary(metrics: dict) -> None:
    print("\nIMAGE-AWARE MLLM JUDGE VS HUMAN")
    print("=" * 72)
    print(f"{'Group':<15} {'N':>4} {'Acc':>9} {'Prec':>9} {'Recall':>9} {'F1':>9} {'Kappa':>8}  CM")
    print("-" * 72)
    for name, m in metrics.items():
        print(
            f"{name:<15} {m['N']:>4} "
            f"{format_pct(m['accuracy']):>9} {format_pct(m['precision']):>9} "
            f"{format_pct(m['recall']):>9} {format_pct(m['f1']):>9} "
            f"{m['cohens_kappa'] if m['cohens_kappa'] is not None else 'N/A':>8}  "
            f"TP={m['TP']}, FP={m['FP']}, TN={m['TN']}, FN={m['FN']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image-aware MLLM judge on human subset")
    parser.add_argument("--annotation", type=Path, default=Path("data/annotations/human_annotation_all_merged.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mllm_judge_human_subset/mllm_judge_human_subset.jsonl"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/mllm_judge_human_subset/agreement_metrics.json"))
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    models_cfg, prompts_cfg, _ = load_configs()
    judge_cfg = models_cfg.get("judge_mllm", models_cfg.get("closed_source"))
    template = prompts_cfg["judge_image_zero_shot"]
    judge = create_image_judge(judge_cfg)

    rows = load_jsonl(args.annotation)
    if args.limit is not None:
        rows = rows[: args.limit]

    completed: dict[str, dict] = {}
    if args.output.exists() and not args.overwrite:
        for row in load_jsonl(args.output):
            annot_id = row.get("annotation_id")
            if annot_id:
                completed[annot_id] = row

    results = []
    for i, row in enumerate(rows, 1):
        annot_id = row.get("annotation_id", f"row_{i}")
        if annot_id in completed:
            results.append(completed[annot_id])
            print(f"[{i}/{len(rows)}] skip existing {annot_id}")
            continue

        out = dict(row)
        prompt = fill_prompt(template, row)
        image_path = Path(str(row.get("image_path", "")))
        image_used = image_path.exists()
        print(f"[{i}/{len(rows)}] image-aware judging {annot_id} ({row.get('task')}, image_used={image_used})")

        try:
            raw = judge.judge(image_path if image_used else None, prompt)
            parsed = parse_judge_json(raw)
        except Exception as exc:
            raw = f"ERROR: {exc}"
            parsed = {
                "hallucination": None,
                "type": "judge_error",
                "confidence": 0.0,
                "image_evidence": "",
                "brief_reason": str(exc),
            }
            print(f"  ERROR: {exc}")

        out["mllm_judge_model_name"] = judge_cfg.get("model_name")
        out["mllm_judge_method"] = "image_zero_shot"
        out["mllm_judge_image_used"] = image_used
        out["mllm_judge_hallucination"] = parsed.get("hallucination")
        out["mllm_judge_hallucination_type"] = parsed.get("type", "unknown")
        out["mllm_judge_confidence"] = parsed.get("confidence", 0.0)
        out["mllm_judge_image_evidence"] = parsed.get("image_evidence", "")
        out["mllm_judge_reason"] = parsed.get("brief_reason", "")
        out["mllm_judge_raw_response"] = raw
        results.append(out)
        save_jsonl(args.output, results)
        time.sleep(args.sleep)

    metrics = grouped_agreement(results)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print_summary(metrics)
    print(f"\nSaved judged rows: {args.output}")
    print(f"Saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
