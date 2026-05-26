# Reroute: Recoverable Visual Token Routing

Codebase for reproducing the paper's main table
(LLaVA-1.5-7B and Qwen2.5-VL-7B on POPE / VQA / RefCOCO) and the
efficiency analysis (TFLOPs + KV cache).

## Quick links

- Main table reproduction → `scripts/run_paper_table.sh`
- Efficiency profiling → `scripts/run_profile.sh` (uses `profiler/`)

## Scope

- LLaVA-1.5-7B (HF `llava-hf/llava-1.5-7b-hf`) and Qwen2.5-VL-7B
  (HF `Qwen/Qwen2.5-VL-7B-Instruct`), single GPU
- Two routing methods: **FastV** (single-shot scoring) and
  **PDrop** (multi-stage cascade, default for reroute)
- Three routing actions:
  - `physical_delete` — vanilla FastV / PDrop behavior (unselected
    tokens are permanently removed from the sequence)
  - `compact_route` — our reroute method: at each routing layer the
    selected K tokens undergo K×K attention while unselected tokens
    bypass via the residual stream, remaining eligible for re-selection
    at later layers
  - `compact_route_stagewise` — same routing decisions as
    `compact_route`, but the sequence stays compact across in-stage
    non-routing layers (bit-identical accuracy, smaller memory bandwidth)
- 38 configs across 3 FLOPs tiers using average token convention:
  - **avg_T = 192**
  - **avg_T = 128**
  - **avg_T = 64**

  Per tier each model has 6 routing variants:
  | Variant | Method | Action | Schedule |
  |---|---|---|---|
  | `fastv_K3` | FastV | physical_delete | single decision @ L3, k=keep_ratio |
  | `pdrop_earlyL2` | PDrop | physical_delete | 4-stage cascade, monotonic |
  | `ours_vs_fastv` | PDrop sched. | compact_route | FLOPs-matched to fastv_K3, uniform keep |
  | `ours_vs_pdrop` | PDrop sched. | compact_route | FLOPs-matched to pdrop_earlyL2 |
  | `ours_vs_fastv_stagewise` | PDrop sched. | compact_route_stagewise | runtime variant of ours_vs_fastv |
  | `ours_vs_pdrop_stagewise` | PDrop sched. | compact_route_stagewise | runtime variant of ours_vs_pdrop |
- Eval harness: POPE + GQA + MMBench + MME + RefCOCO/+/g (8 splits)
- TFLOPs + KV cache profiler

## Repo layout

```
reroute_release/
├── README.md
├── requirements.txt
├── lmms_eval_patches/        # apply.sh + 3 utils_rec.py patches
├── configs/
│   ├── eval/                 # per-benchmark eval configs (pope, gqa, mme, mmbench, refcoco{,+,g}*)
│   ├── model/                # model-side configs (llava15_7b_hf, qwen25vl_7b)
│   └── experiment/
│       ├── baseline/         # llava15.yaml, qwen25vl.yaml (no routing)
│       ├── llava15/
│       │   ├── avg192/       # 6 routing variants
│       │   ├── avg128/       # 6 routing variants
│       │   └── avg64/        # 6 routing variants
│       └── qwen25vl/
│           ├── avg192/, avg128/, avg64/   (6 variants each)
├── models/                   # router + patching dispatcher
├── scripts/                  # run_eval, run_setting, run_paper_table, run_profile
├── profiler/                 # TFLOPs + KV-cache measurement
└── experiments/              # output logs land here (gitignored)
```

Each experiment config is `experiment/<model>/<tier>/<model>_<variant>_<tier>.yaml`.
For example, `experiment/llava15/avg192/llava15_ours_vs_fastv_avg192.yaml`.

## Env

`requirements.txt` pins the Python deps. `lmms-eval` is installed
**editable from the upstream `v0.7.1` git tag**.

```
torch==2.11.0+cu128       # CUDA 12.8; see requirements.txt for other CUDA
transformers==5.4.0
hydra-core==1.3.2
lmms-eval @ v0.7.1        # editable install from GitHub tag
```

After installing lmms-eval, run `bash lmms_eval_patches/apply.sh` once
to overlay the 3 `utils_rec.py` patches needed for Qwen2.5-VL RefCOCO
reproduction (adds `BBOX_COORD_FORMAT` and `REFCOCO_PROMPT_STYLE` env
vars plus the matching RefCOCO prompt branch).

## Run

```bash
# 1. Create env
conda create -n reroute_release python=3.10 -y && conda activate reroute_release

# 2. Install lmms-eval (editable from upstream v0.7.1 tag)
git clone --branch v0.7.1 --depth 1 \
    https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e ./lmms-eval

# 3. Install the rest of the deps
pip install -r requirements.txt

# 4. Overlay the 3 utils_rec.py patches (idempotent)
bash lmms_eval_patches/apply.sh
# Verify:
#   python -c 'from lmms_eval.tasks import TaskManager; TaskManager(); print("OK")'

# Optional: deepspeed for TFLOPs profiling. Requires nvcc.
#   pip install deepspeed==0.14.5
# Accuracy reproduction does NOT require deepspeed.

# 5. Reproduce the main table — pick model and tier
bash scripts/run_paper_table.sh llava15                 # 7 configs @ avg192 (default)
bash scripts/run_paper_table.sh qwen25vl                # 7 configs @ avg192
bash scripts/run_paper_table.sh all                     # 14 configs (both models, avg192)
bash scripts/run_paper_table.sh llava15  --tier avg128  # 7 configs @ avg128
bash scripts/run_paper_table.sh all      --tier all     # 38 configs (both models, 3 tiers)

# By default each dispatched config runs POPE only. To pick the eval set:
bash scripts/run_paper_table.sh llava15 --tasks vqa         # gqa + mmbench + mme
bash scripts/run_paper_table.sh llava15 --tasks refcoco     # 3 RefCOCO splits
bash scripts/run_paper_table.sh llava15 --tasks grounding   # 8 RefCOCO/+/g splits
bash scripts/run_paper_table.sh llava15 --tasks ablation    # gqa+mmbench+2 RefCOCO

# 6. Efficiency profile (TFLOPs + KV cache, GPU time)
bash scripts/run_profile.sh llava15
bash scripts/run_runtime_bench.sh llava15
```

## Acknowledgements

This repository builds upon [LLaVA](https://github.com/haotian-liu/LLaVA),
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), and
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). 
We also thank
the authors of [FastV](https://github.com/pkunlp-icler/FastV),
[PDrop](https://github.com/Cooperx521/PyramidDrop),
[p-MoD](https://github.com/MCG-NJU/p-MoD), and
[NuWA](https://github.com/Man-PaperRejected/Nuwa) for their inspiring work and
publicly available resources.
