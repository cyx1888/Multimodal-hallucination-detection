"""
Shared config loader: reads YAML configs and applies environment variable overrides.

Environment variable mapping:
  FOURROUTER_API_KEY  -> closed_source.api_key, judge.api_key
  FOURROUTER_API_BASE -> closed_source.api_base, judge.api_base
  SILICONFLOW_API_KEY -> open_source.api_key
  SILICONFLOW_API_BASE -> open_source.api_base

Usage:
  from config_loader import load_configs
  models, prompts, eval_cfg = load_configs()
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Auto-load .env from project root
_ENV_LOADED = False


def _ensure_env():
    global _ENV_LOADED
    if not _ENV_LOADED:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        _ENV_LOADED = True


def _apply_env_overrides(models_cfg: dict) -> dict:
    """Apply environment variable overrides to model config."""
    # 4router keys (used by both closed_source and judge)
    fourrouter_key = os.environ.get("FOURROUTER_API_KEY", "")
    fourrouter_base = os.environ.get("FOURROUTER_API_BASE", "")

    # SiliconFlow keys (used by open_source)
    siliconflow_key = os.environ.get("SILICONFLOW_API_KEY", "")
    siliconflow_base = os.environ.get("SILICONFLOW_API_BASE", "")

    # Apply to closed_source
    if "closed_source" in models_cfg:
        if fourrouter_key:
            models_cfg["closed_source"]["api_key"] = fourrouter_key
        if fourrouter_base:
            models_cfg["closed_source"]["api_base"] = fourrouter_base

    # Apply to open_source
    if "open_source" in models_cfg:
        if siliconflow_key:
            models_cfg["open_source"]["api_key"] = siliconflow_key
        if siliconflow_base:
            models_cfg["open_source"]["api_base"] = siliconflow_base

    # Apply to judges (same 4router keys)
    for judge_key in ("judge", "judge_mllm"):
        if judge_key in models_cfg:
            if fourrouter_key:
                models_cfg[judge_key]["api_key"] = fourrouter_key
            if fourrouter_base:
                models_cfg[judge_key]["api_base"] = fourrouter_base

    return models_cfg


def load_configs():
    """Load all config files with env var overrides."""
    _ensure_env()
    config_dir = Path(__file__).parent.parent / "configs"

    with open(config_dir / "models.yaml", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    with open(config_dir / "prompts.yaml", encoding="utf-8") as f:
        prompts_cfg = yaml.safe_load(f)

    models_cfg = _apply_env_overrides(models_cfg)

    # eval.yaml is optional
    eval_cfg = {}
    eval_path = config_dir / "eval.yaml"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            eval_cfg = yaml.safe_load(f)

    return models_cfg, prompts_cfg, eval_cfg


def load_models_config():
    """Load only models config with env overrides."""
    models_cfg, _, _ = load_configs()
    return models_cfg
