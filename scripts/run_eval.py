"""Run evaluation with visual token routing.

Usage:
    conda run -n vlm_routing_4070ti python scripts/run_eval.py \
        model=llava_7b routing=fastv routing.action=residual_skip

This script:
1. Loads the VLM via lmms-eval's model class
2. Patches it with our routing mechanism
3. Runs lmms-eval benchmarks
4. Logs results to wandb and local results file
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path (before hydra changes cwd)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import os
from datetime import datetime

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from models.router import FastVRouter, PDropRouter
from models.dispatcher import TokenDispatcher
from models.patching import patch_model_for_routing, get_visual_token_finder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_router(cfg: DictConfig):
    """Construct a router from config. Release supports FastV and PDrop."""
    if cfg.routing.method == "fastv":
        keep_ratio = (cfg.routing.keep_ratios[0]
                      if "keep_ratios" in cfg.routing
                      else cfg.routing.keep_ratio)
        return FastVRouter(
            scoring_layer=cfg.routing.scoring_layer,
            keep_ratio=keep_ratio,
        )
    elif cfg.routing.method == "pdrop":
        return PDropRouter(
            drop_layers=list(cfg.routing.drop_layers),
            keep_ratios=list(cfg.routing.keep_ratios),
            monotonic=cfg.routing.get("monotonic", True),
        )
    elif cfg.routing.method == "none":
        return None
    else:
        raise ValueError(
            f"Unknown routing method: {cfg.routing.method}. "
            f"Release supports: fastv, pdrop, none."
        )


def get_model_family(model_name: str) -> str:
    """Determine model family from model name."""
    name = model_name.lower()
    if "llava" in name:
        return "llava"
    elif "qwen3-vl" in name or "qwen3_vl" in name:
        return "qwen3vl"
    elif "qwen" in name:
        return "qwen25vl"
    else:
        raise ValueError(f"Unknown model family for: {model_name}")


def run_lmms_eval(cfg: DictConfig) -> dict:
    """Run lmms-eval with the patched model.

    Uses lmms-eval's Python API to load a model, patch it, and evaluate.
    """
    from lmms_eval import evaluator
    from lmms_eval.models import get_model

    model_name = cfg.model.pretrained
    model_family = get_model_family(model_name)
    lmms_model_type = cfg.model.lmms_model_type  # "llava_hf" or "qwen2_5_vl"

    logger.info(f"Loading model: {model_name} (type: {lmms_model_type})")

    # Build model_args string for lmms-eval
    model_args = f"pretrained={model_name},device_map=auto"
    if cfg.model.get("attn_implementation"):
        model_args += f",attn_implementation={cfg.model.attn_implementation}"
    if cfg.model.get("dtype"):
        model_args += f",dtype={cfg.model.dtype}"
    if cfg.model.get("max_pixels"):
        model_args += f",max_pixels={cfg.model.max_pixels}"
    if cfg.model.get("enable_thinking") is not None:
        model_args += f",enable_thinking={cfg.model.enable_thinking}"

    # Load model via lmms-eval's standard API
    model_cls = get_model(lmms_model_type)
    lm = model_cls.create_from_arg_string(
        model_args,
        {"batch_size": cfg.eval.batch_size},
    )

    # Patch the model with routing
    router = build_router(cfg)
    if router is not None:
        dispatcher = TokenDispatcher()
        action = cfg.routing.action
        logger.info(f"Patching model with routing: method={cfg.routing.method}, "
                     f"action={action}")

        ctx = patch_model_for_routing(
            lm._model,
            router=router,
            dispatcher=dispatcher,
            action=action,
            model_family=model_family,
        )

        # Store context on the lmms model wrapper for access during generation
        lm._routing_ctx = ctx
    else:
        logger.info("No routing — running baseline evaluation")

    # Per-model-family RefCOCO evaluation presets.
    # bbox_format and prompt_style are paired — each family is SFT'd on a
    # specific (prompt distribution, output coord format) combo, and mixing
    # them across families silently produces 0% ACC (Gate-0.4 lesson). Keep
    # them in ONE dict so any new family adds both atomically or not at all.
    #
    # bbox format:
    #   "normalized" = already [0,1]       (LLaVA-1.5 SFT target)
    #   "pixel"      = divide by image dims (Qwen2.5-VL smart_resize pixels)
    #   "scale_1000" = divide by 1000      (Qwen3-VL normalized ints)
    #
    # prompt style:
    #   "full"  = lmms-eval canonical Shikra/KOSMOS-2 prompt, "[0,1] bounded"
    #   "short" = trimmed, just the ask + caption (no format spec)
    #   "nuwa"  = "Locate X ... JSON format." — Nuwa-paper Qwen prompt
    #             (Gate-0.4 external evidence: Qwen2.5-VL 84% ACC@0.5 @ k=0.25
    #              under this prompt; full → 0% in same harness)
    #
    # Env vars are the sole transport because lmms-eval loads task utils via
    # importlib.spec_from_file_location, creating a separate module instance
    # from the standard import path — process-level globals won't reach it.
    MODEL_PRESETS = {
        "llava":    {"bbox": "normalized", "prompt": "full"},
        "qwen25vl": {"bbox": "pixel",      "prompt": "nuwa"},
        "qwen3vl":  {"bbox": "scale_1000", "prompt": "nuwa"},
        "qwen35":   {"bbox": "scale_1000", "prompt": "nuwa"},
    }
    preset = MODEL_PRESETS.get(model_family, {"bbox": "auto", "prompt": "full"})

    # Override order for prompt_style AND bbox_format:
    #   1. Shell env REFCOCO_PROMPT_STYLE / BBOX_COORD_FORMAT (A/B runs, verify scripts)
    #   2. Hydra cfg.eval.prompt_style (per-experiment override; prompt only)
    #   3. family default from MODEL_PRESETS
    # bbox_format override added 2026-04-22 for verify_refcoco_root_cause.sh —
    # without it, downstream env overrides get silently stomped by preset.
    cfg_style = cfg.eval.get("prompt_style", None)
    env_style = os.environ.get("REFCOCO_PROMPT_STYLE")
    prompt_style = env_style or cfg_style or preset["prompt"]
    prompt_source = "env" if env_style else ("cfg" if cfg_style else "default")

    env_bbox = os.environ.get("BBOX_COORD_FORMAT")
    bbox_format = env_bbox or preset["bbox"]
    bbox_source = "env" if env_bbox else "default"

    os.environ["BBOX_COORD_FORMAT"] = bbox_format
    os.environ["REFCOCO_PROMPT_STYLE"] = prompt_style
    logger.info(
        f"RefCOCO preset (model_family={model_family}): "
        f"bbox={bbox_format} (source={bbox_source}), "
        f"prompt={prompt_style} (source={prompt_source})"
    )

    # Run evaluation
    tasks = list(cfg.eval.benchmarks)
    logger.info(f"Evaluating on: {tasks}")

    eval_kwargs = dict(
        model=lm,
        tasks=tasks,
        batch_size=cfg.eval.batch_size,
        log_samples=cfg.eval.get("log_samples", False),
    )
    limit = cfg.eval.get("limit", None)
    if limit is not None:
        eval_kwargs["limit"] = limit

    results = evaluator.simple_evaluate(**eval_kwargs)

    return results


def log_results(cfg: DictConfig, results: dict) -> None:
    """Log results to console and local file."""
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    result_summary = {
        "model": cfg.model.pretrained,
        "routing_method": cfg.routing.method,
        "routing_action": cfg.routing.get("action", "none"),
        "keep_ratio": cfg.routing.get("keep_ratio", None) or list(cfg.routing.get("keep_ratios", [1.0])),
        "timestamp": datetime.now().isoformat(),
    }

    if "results" in results:
        for task_name, task_results in results["results"].items():
            logger.info(f"\n{task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
                    result_summary[f"{task_name}/{metric}"] = value

    # Per-experiment artifact dir = Hydra's runtime output_dir.
    # Hydra 1.2+ default is chdir=false, so relative Path("results.json")
    # would land in launch-CWD (project root) and get clobbered across arms.
    # Resolve against HydraConfig explicitly → each arm's files stay isolated.
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        run_dir = Path.cwd()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Annotate the summary with run_dir so the global JSONL can be back-traced.
    result_summary["run_dir"] = str(run_dir)

    local_results = run_dir / "results.json"
    with open(local_results, "w") as f:
        json.dump(result_summary, f, indent=2)
    logger.info(f"Results saved to {local_results}")

    global_results = _PROJECT_ROOT / "experiments" / "eval_results.jsonl"
    with open(global_results, "a") as f:
        f.write(json.dumps(result_summary) + "\n")
    logger.info(f"Appended to {global_results}")

    # Per-sample outputs (model responses) if log_samples was enabled
    if "samples" in results:
        samples_file = run_dir / "samples.json"
        with open(samples_file, "w") as f:
            json.dump(results["samples"], f, indent=2, default=str)
        logger.info(f"Per-sample outputs saved to {samples_file}")


@hydra.main(config_path=str(_PROJECT_ROOT / "configs"), config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if OmegaConf.select(cfg, "hydra.verbose", default=False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    results = run_lmms_eval(cfg)
    log_results(cfg, results)


if __name__ == "__main__":
    main()
