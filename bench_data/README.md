# Bench cohort

3 COCO val 2014 images + describe prompt. Source: `lmms-lab/POPE` (test split), CC-BY 4.0. Used by both `profiler/bench_runtime.py` (wall-clock) and `profiler/measure_metrics.py` (prefill TFLOPs + KV).

| sample | image_id | size |
|---|---|---|
| 0 | `COCO_val2014_000000310196` | 640×427 |
| 1 | `COCO_val2014_000000210789` | 369×520 |
| 2 | `COCO_val2014_000000429109` | 640×427 |

Prompt: `"Describe this image in detail."`

`manifest.json` carries `(sample_id, image_id, category, question, image, image_WxH)` per sample.

## Run — runtime bench (wall-clock)

```bash
bash scripts/run_runtime_bench.sh llava15                                # 7 configs @ avg192
bash scripts/run_runtime_bench.sh qwen25vl --tier avg64                  # 7 configs @ avg64
bash scripts/run_runtime_bench.sh all --tier all                         # 44 configs

bash scripts/run_runtime_bench.sh llava15 --tier avg192 \
    --n-passes 10 --n-warmup-passes 3 --n-decode-tokens 128 --gpu 0
```

Single config:

```bash
python profiler/bench_runtime.py \
    --config-name experiment/baseline/llava15 \
    --n-passes 5 --n-warmup-passes 2 --n-decode-tokens 64 \
    --out experiments/runtime/llava15_baseline.json
```

`profiler/bench_runtime.py` flags:
- `--config-name` (required): Hydra config path under `configs/`, no `.yaml`.
- `--n-passes` (default 3): timed passes through the cohort.
- `--n-warmup-passes` (default 1): untimed warmup passes.
- `--n-decode-tokens` (default 64): forced generation length.
- `--out` (required): output JSON path.

`run_runtime_bench.sh` flags:
- `<model>` (positional): `llava15` | `qwen25vl` | `all`.
- `--tier` (default `avg192`): `avg192` | `avg128` | `avg64` | `all`.
- `--n-passes` (default 5), `--n-warmup-passes` (default 2), `--n-decode-tokens` (default 64), `--gpu <id>`.

## Run — prefill profiler (TFLOPs + KV)

```bash
bash scripts/run_profile.sh llava15                                      # 7 configs @ avg192
bash scripts/run_profile.sh all --tier all                               # 44 configs
```

Single config:

```bash
python profiler/measure_metrics.py \
    --config-name experiment/baseline/llava15 \
    --n-passes 3 --n-warmup-passes 1 \
    --out experiments/profile/llava15_baseline.json
```

`measure_metrics.py` flags:
- `--config-name` (required), `--n-passes` (default 3), `--n-warmup-passes` (default 1), `--out` (required).
- Scope is prefill-only; no decode flag.

`run_profile.sh` flags:
- `<model>` (positional): `llava15` | `qwen25vl` | `all`.
- `--tier` (default `avg192`), `--n-passes` (default 3), `--n-warmup-passes` (default 1), `--gpu <id>`.

## Replacing the cohort

Overwrite this directory with your own samples + manifest:

```json
[
  {
    "sample_id":   0,
    "image_id":    "any string identifier",
    "category":    "any string label",
    "question":    "Describe this image in detail.",
    "image":       "sample_00.jpg",
    "image_WxH":   [640, 427]
  }
]
```
