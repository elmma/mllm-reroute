#!/usr/bin/env python3
"""Prefill-only TFLOPs + KV-cache profiler.

Reads the bundled cohort at `bench_data/`, runs a single prefill
forward (no decode) per sample × n_passes, captures DeepSpeed TFLOPs
around the prefill call and walks the post-prefill KV cache.

Scope is intentionally prefill-only: routing fires during prefill,
decode steps would dilute the routing-attributable signal.

Usage:
  python profiler/measure_metrics.py \
      --config-name experiment/llava15/avg192/llava15_ours_vs_fastv_stagewise_avg192 \
      --n-passes 3 --n-warmup-passes 1 \
      --out experiments/profile/llava15_ours_vs_fastv_stagewise_avg192.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Allow imports from release root (../models)
_RELEASE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RELEASE_ROOT))

def _build_model_and_apply_routing(cfg_path: Path):
    """Load HF model (LLaVA or Qwen2.5-VL) + apply routing patch."""
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from models.router import FastVRouter, PDropRouter
    from models.dispatcher import TokenDispatcher
    from models.patching import patch_model_for_routing

    cfg = OmegaConf.load(cfg_path)

    # Parse the experiment yaml's `defaults:` list to find the model
    # config name (e.g., "llava15_7b_hf", "qwen25vl_7b").
    # OmegaConf produces DictConfig elements here, not dict, so use
    # hasattr / OmegaConf-aware checks.
    model_name = "llava15_7b_hf"
    for d in cfg.get("defaults", []):
        if hasattr(d, "keys") and "/model" in d:
            model_name = str(d["/model"])
            break
    model_cfg = OmegaConf.load(_RELEASE_ROOT / "configs" / "model" / f"{model_name}.yaml")

    pretrained = model_cfg.pretrained
    lmms_type = model_cfg.get("lmms_model_type", "llava_hf")
    if lmms_type == "llava_hf":
        from transformers import LlavaForConditionalGeneration as ModelCls
        model_family = "llava"
    elif lmms_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        model_family = "qwen25vl"
    elif lmms_type in ("qwen3_vl", "qwen3vl"):
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
        model_family = "qwen3vl"
    else:
        raise ValueError(f"Unknown lmms_model_type: {lmms_type}")

    # Loading: LLaVA → fp16 + .to(cuda); Qwen2.5-VL/Qwen3-VL → bfloat16
    # + device_map="auto" (matches lmms-eval's qwen loader defaults).
    dtype = torch.float16 if model_family == "llava" else torch.bfloat16
    print(f"[PROFILER] loading {pretrained} (dtype={dtype}, family={model_family}) ...", flush=True)
    load_kwargs = dict(
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    if model_family == "llava":
        model = ModelCls.from_pretrained(pretrained, **load_kwargs).to("cuda").eval()
    else:
        # Qwen2.5-VL / Qwen3-VL : let HF place modules; bfloat16 keeps
        # within 16 GB local for the 7B variant.
        load_kwargs["device_map"] = "auto"
        model = ModelCls.from_pretrained(pretrained, **load_kwargs).eval()
    proc_kwargs = {}
    if model_cfg.get("max_pixels"):
        proc_kwargs["max_pixels"] = int(model_cfg.max_pixels)
    processor = AutoProcessor.from_pretrained(pretrained, **proc_kwargs)

    # Build router according to cfg.routing
    r_cfg = cfg.routing
    if r_cfg.method == "none":
        print("[PROFILER] no routing — vanilla baseline", flush=True)
    elif r_cfg.method == "fastv":
        router = FastVRouter(
            scoring_layer=r_cfg.scoring_layer,
            keep_ratio=r_cfg.keep_ratio,
        )
        dispatcher = TokenDispatcher()
        patch_model_for_routing(
            model, router, dispatcher,
            action=r_cfg.action,
            
            model_family=model_family,
        )
        print(f"[PROFILER] patched with FastV K={r_cfg.scoring_layer} "
              f"keep={r_cfg.keep_ratio} action={r_cfg.action}", flush=True)
    elif r_cfg.method == "pdrop":
        router = PDropRouter(
            drop_layers=list(r_cfg.drop_layers),
            keep_ratios=list(r_cfg.keep_ratios),
            monotonic=r_cfg.get("monotonic", False),
        )
        dispatcher = TokenDispatcher()
        patch_model_for_routing(
            model, router, dispatcher,
            action=r_cfg.action,
            
            model_family=model_family,
        )
        print(f"[PROFILER] patched with PDrop drops={r_cfg.drop_layers} "
              f"keeps={r_cfg.keep_ratios} action={r_cfg.action}", flush=True)
    else:
        raise ValueError(f"Unknown routing.method: {r_cfg.method}")

    return model, processor

BENCH_BUNDLED = _RELEASE_ROOT / "bench_data"


def _load_bundled_cohort() -> tuple[Path, list[dict]]:
    manifest_p = BENCH_BUNDLED / "manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(
            f"Bundled bench cohort not found at {BENCH_BUNDLED}. "
            f"Re-extract the release tarball."
        )
    manifest = json.loads(manifest_p.read_text())
    for s in manifest:
        if not (BENCH_BUNDLED / s["image"]).exists():
            raise FileNotFoundError(f"Manifest references missing image: {BENCH_BUNDLED / s['image']}")
    return BENCH_BUNDLED, manifest


def _build_inputs_for_sample(processor, model, sample: dict, bench_dir: Path):
    from PIL import Image
    img = Image.open(bench_dir / sample["image"]).convert("RGB")
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": sample["question"]},
    ]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(
        model.device, dtype=torch.float16 if model.dtype == torch.float16 else torch.bfloat16
    )
    inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()
    return inputs

def _walk_kv(out):
    """Sum K+V tokens and bytes across layers; return per-layer K seq lengths."""
    n_tokens, n_bytes, per_layer = 0, 0, []
    kv = out.past_key_values
    if hasattr(kv, "key_cache") and hasattr(kv, "value_cache"):
        iters = zip(kv.key_cache, kv.value_cache)
    else:
        iters = ((lk[0], lk[1]) for lk in kv)
    for k, v in iters:
        n_tokens += k.shape[-2] + v.shape[-2]
        n_bytes  += k.numel() * k.element_size() + v.numel() * v.element_size()
        per_layer.append(int(k.shape[-2]))
    return n_tokens, n_bytes, per_layer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True,
                    help="release experiment config path (e.g. experiment/baseline/llava15)")
    ap.add_argument("--n-passes", type=int, default=3,
                    help="timed prefill passes through the bundled cohort")
    ap.add_argument("--n-warmup-passes", type=int, default=1,
                    help="untimed warmup passes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_path = _RELEASE_ROOT / "configs" / f"{args.config_name}.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr); return 1

    bench_dir, manifest = _load_bundled_cohort()
    print(f"[PROFILER] cohort source: {bench_dir} ({len(manifest)} samples)", flush=True)

    model, processor = _build_model_and_apply_routing(cfg_path)
    per_sample_inputs = [{
        "sample": s,
        "inputs": _build_inputs_for_sample(processor, model, s, bench_dir),
    } for s in manifest]
    for entry in per_sample_inputs:
        entry["prefill_tokens"] = int(entry["inputs"]["input_ids"].shape[1])

    try:
        from deepspeed.profiling.flops_profiler import FlopsProfiler
        prof = FlopsProfiler(model)
        deepspeed_available = True
    except Exception as e:
        prof, deepspeed_available = None, False
        print(f"[WARN] deepspeed unavailable ({type(e).__name__}: {e}) — "
              f"TFLOPs will be null. To enable: `conda install -c nvidia cuda-nvcc=12.1` "
              f"then re-run.", flush=True)

    # Warmup passes (untimed) — single prefill forward, no decode
    print(f"[PROFILER] warmup {args.n_warmup_passes} pass × {len(per_sample_inputs)} samples ...", flush=True)
    for _ in range(args.n_warmup_passes):
        for entry in per_sample_inputs:
            with torch.inference_mode():
                _ = model(**entry["inputs"], use_cache=True)
    torch.cuda.empty_cache()

    # Timed passes — measure PREFILL ONLY (TFLOPs + post-prefill KV).
    # Decode is excluded because routing fires only at prefill; decode
    # FLOPs/KV would dilute the routing-attributable signal.
    per_sample_records = [{
        "sample_id": entry["sample"]["sample_id"],
        "image_id":  entry["sample"].get("image_id"),
        "category":  entry["sample"].get("category"),
        "image_WxH": entry["sample"]["image_WxH"],
        "prefill_tokens": entry["prefill_tokens"],
        "tflops": [],
        "kv_cache_tokens": [],
        "kv_cache_bytes": [],
        "per_layer_kv_seq_len": None,
    } for entry in per_sample_inputs]

    print(f"[PROFILER] timed {args.n_passes} prefill passes × {len(per_sample_inputs)} samples = "
          f"{args.n_passes * len(per_sample_inputs)} measurements", flush=True)
    for p in range(args.n_passes):
        for i, entry in enumerate(per_sample_inputs):
            if prof is not None: prof.start_profile()
            with torch.inference_mode():
                out = model(**entry["inputs"], use_cache=True)
            if prof is not None:
                prof.stop_profile()
                per_sample_records[i]["tflops"].append(prof.get_total_flops() / 1e12)
                prof.reset_profile()
            n_tok, n_bytes, per_layer = _walk_kv(out)
            per_sample_records[i]["kv_cache_tokens"].append(n_tok)
            per_sample_records[i]["kv_cache_bytes"].append(n_bytes)
            per_sample_records[i]["per_layer_kv_seq_len"] = per_layer
            del out
            torch.cuda.empty_cache()
    if prof is not None: prof.end_profile()

    # Per-sample medians
    for rec in per_sample_records:
        if rec["tflops"]:
            rec["tflops_median"] = float(sorted(rec["tflops"])[len(rec["tflops"])//2])
        rec["kv_cache_tokens_median"] = int(sorted(rec["kv_cache_tokens"])[len(rec["kv_cache_tokens"])//2])
        rec["kv_cache_MB_median"]     = float(sorted(rec["kv_cache_bytes"])[len(rec["kv_cache_bytes"])//2]) / 2**20

    # Cohort aggregate
    all_tflops = [t for rec in per_sample_records for t in rec["tflops"]]
    all_kv_tok = [t for rec in per_sample_records for t in rec["kv_cache_tokens"]]
    all_kv_bytes = [b for rec in per_sample_records for b in rec["kv_cache_bytes"]]
    def _mean(xs): return (sum(xs) / len(xs)) if xs else None
    aggregate = {
        "prefill_tflops_mean":         _mean(all_tflops),
        "prefill_kv_cache_tokens_mean":_mean(all_kv_tok),
        "prefill_kv_cache_MB_mean":    (_mean(all_kv_bytes) / 2**20) if all_kv_bytes else None,
    }

    summary = {
        "config": args.config_name,
        "scope": "prefill_only",
        "bench_data": {
            "source": "COCO val 2014 (via lmms-lab/POPE images)",
            "prompt": "Describe this image in detail.",
            "cohort_size": len(per_sample_inputs),
            "cohort_dir": str(bench_dir),
        },
        "settings": {
            "n_warmup_passes": args.n_warmup_passes,
            "n_passes": args.n_passes,
            "deepspeed_available": deepspeed_available,
        },
        "per_sample": per_sample_records,
        "aggregate": aggregate,
    }

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] wrote {out_p}  (scope: prefill_only)", flush=True)
    if aggregate["prefill_tflops_mean"] is not None:
        print(f"  prefill TFLOPs/sample (mean): {aggregate['prefill_tflops_mean']:.3f}", flush=True)
    print(f"  prefill KV cache tokens (mean): {aggregate['prefill_kv_cache_tokens_mean']:.0f}", flush=True)
    print(f"  prefill KV cache MB     (mean): {aggregate['prefill_kv_cache_MB_mean']:.2f}", flush=True)
    for rec in per_sample_records:
        per_layer = rec["per_layer_kv_seq_len"]
        print(f"  sample_{rec['sample_id']} ({len(per_layer)} layers): {per_layer}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
