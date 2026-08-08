# MLLM Hallucination Detection and Evaluation

This repository contains an inference-only evaluation pipeline for detecting and analyzing hallucinations in multimodal large language models (MLLMs). The experiments compare a closed-source model (`gpt-5.4`) and an open-source model (`Qwen/Qwen3-VL-8B-Instruct`) on visual question answering, visual mathematical reasoning, and an OCR/text-rich reasoning extension.

The repository is organized as a reproducible code package. Final report and slide deliverables are kept outside this repository; generated documents, raw downloads, raw model responses, caches, and local credentials are excluded.

## Evaluation Scope

| Task | Role | Samples | Focus |
| --- | --- | ---: | --- |
| POPE | Required VQA task | 300 | Object-existence VQA and object hallucination |
| MathVista | Required secondary task | 100 | Visual mathematical reasoning |
| LogicOCR | Bonus extension | 50 | OCR/text-rich logical reasoning |

## Models

| Model | Category | Access |
| --- | --- | --- |
| `gpt-5.4` | Closed-source | OpenAI-compatible API |
| `Qwen/Qwen3-VL-8B-Instruct` | Open-source | OpenAI-compatible API |

## Method

- Direct prompting and chain-of-thought prompting are compared for each task/model pair.
- POPE additionally uses a rule-based object hallucination detector: ground truth `no` with model answer `yes` is counted as object hallucination.
- All tasks use an image-aware MLLM judge for hallucination classification.
- Human-as-judge validation is measured on 40 main-task samples plus 20 LogicOCR samples.
- Analysis covers model comparison, CoT impact, hallucination type distribution, judge-human agreement, and representative failure cases.

## Repository Layout

```text
.
  configs/                 Model, prompt, and evaluation configuration
  data/
    annotations/           Human labels, merged annotation sets, and guidelines
    sampled/               Sample metadata JSONL files; images are regenerated locally
  outputs/
    metrics/               Final task and agreement metrics
    figures/               Final analysis JSON for charts and failure cases
    mllm_judge_full/       Compact image-aware judge metric summaries
  scripts/                 Data preparation, inference, judging, evaluation, and analysis
```

Excluded from version control:

- `.env` and other credential files
- `download/` and full upstream datasets
- `data/sampled/**/images/`
- raw model-answer JSONL files
- raw judged JSONL files
- generated report/PPT/DOCX artifacts
- cache and IDE files

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a local `.env` from `.env.example` and fill in API credentials:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## Data Preparation

Prepare POPE and MathVista from Hugging Face:

```bash
python scripts/prepare_data.py --pope_n 300 --mathvista_n 100 --seed 42
```

Or prepare them from local parquet files under `download/`:

```bash
python scripts/prepare_data_local.py --pope_n 300 --mathvista_n 100 --seed 42
```

Prepare LogicOCR from local `download/LogicOCR/LogicOCR_gen.json` and `LogicOCR_gen.zip`:

```bash
python scripts/prepare_logicocr.py --n 50 --seed 42
```

## Running Experiments

Run one inference job:

```bash
python scripts/run_inference.py ^
  --task pope ^
  --model_type closed_source ^
  --prompt_mode direct ^
  --input data/sampled/pope/pope_sampled.jsonl ^
  --image_dir data/sampled/pope/images ^
  --output_dir outputs/model_answers
```

Run rule-based and image-aware judging:

```bash
python scripts/run_judge.py ^
  --task pope ^
  --input outputs/model_answers/pope_gpt-5.4_direct.jsonl ^
  --output_dir outputs/judge_results
```

Evaluate a judged file:

```bash
python scripts/evaluate.py ^
  --task pope ^
  --input outputs/judge_results/pope_gpt-5.4_direct_judged.jsonl ^
  --annotation data/annotations/human_annotation_all_merged.jsonl ^
  --output_dir outputs/metrics
```

Run error analysis over judged outputs:

```bash
python scripts/analyze_errors.py ^
  --input_dir outputs/judge_results ^
  --annotation data/annotations/human_annotation_all_merged.jsonl ^
  --output_dir outputs/figures
```

Generate the compact image-aware judge summary after full judged JSONL files are available:

```bash
python scripts/generate_mllm_summary.py
```

## Results Included

The repository includes compact final result files:

- `outputs/metrics/*.json`: task metrics and human-agreement metrics
- `outputs/figures/*.json`: CoT impact, type distribution, human disagreements, and failure cases
- `outputs/mllm_judge_full/full_mllm_judge_summary.*`: full image-aware judge summary tables

These files are small enough for review and preserve the key experimental evidence without committing raw API responses or large datasets.
