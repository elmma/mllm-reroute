#!/bin/bash
# Efficiency profile (TFLOPs + KV cache) — thin dispatcher that loops
# every release config and writes per-config JSONs into experiments/profile/.
#
# Usage:
#   bash scripts/run_profile.sh <model> [--tier <T>] \
#        [--n_samples N] [--max_new_tokens N] [--gpu <id>]
#
# model:
#   llava15     LLaVA-1.5-7B configs for selected tier(s)
#   qwen25vl    Qwen2.5-VL-7B configs for selected tier(s)
#   all         both models
#
# --tier T:  avg192 (default) | avg128 | avg64 | all

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [--tier avg192|avg128|avg64|all] [--n-passes N] [--n-warmup-passes N] [--gpu <id>]}"
shift

TIER="avg192"
N_PASSES=1
N_WARMUP=0
GPU_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)              TIER="$2"; shift 2 ;;
        --tier=*)            TIER="${1#--tier=}"; shift ;;
        --n-passes)          N_PASSES="$2"; shift 2 ;;
        --n-warmup-passes)   N_WARMUP="$2"; shift 2 ;;
        --gpu)               GPU_ARG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$GPU_ARG" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ARG"; fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PROJECT_ROOT}/experiments/profile"
mkdir -p "$OUT_DIR"

ROUTING_VARIANTS=(
    "fastv_K3"
    "pdrop_earlyL2"
    "ours_vs_fastv"
    "ours_vs_pdrop"
    "ours_vs_fastv_stagewise"
    "ours_vs_pdrop_stagewise"
)

case "$TIER" in
    avg192|avg128|avg64) TIERS=("$TIER") ;;
    all) TIERS=("avg192" "avg128" "avg64") ;;
    *) echo "Unknown tier: $TIER (use: avg192 | avg128 | avg64 | all)" >&2; exit 1 ;;
esac

build_configs() {
    local model="$1"
    local out=("baseline/${model}")
    for t in "${TIERS[@]}"; do
        for v in "${ROUTING_VARIANTS[@]}"; do
            out+=("${model}/${t}/${model}_${v}_${t}")
        done
    done
    printf '%s\n' "${out[@]}"
}

case "$MODEL" in
    llava15)  mapfile -t CONFIGS < <(build_configs "llava15") ;;
    qwen25vl) mapfile -t CONFIGS < <(build_configs "qwen25vl") ;;
    all)
        mapfile -t LLAVA_CFGS < <(build_configs "llava15")
        mapfile -t QWEN_CFGS  < <(build_configs "qwen25vl")
        CONFIGS=("${LLAVA_CFGS[@]}" "${QWEN_CFGS[@]}")
        ;;
    *) echo "Unknown model: $MODEL  (use: llava15 | qwen25vl | all)" >&2; exit 1 ;;
esac

CONDA_ENV="${CONDA_ENV:-reroute_release}"
cd "$PROJECT_ROOT"

echo "[profile dispatcher] tier=$TIER  model=$MODEL  configs=${#CONFIGS[@]}"

for CFG in "${CONFIGS[@]}"; do
    TAG="${CFG//\//_}"          # flatten path → single JSON filename
    OUT="${OUT_DIR}/${TAG}.json"
    echo
    echo "===== $(date '+%F %T'): PROFILE ${CFG} → ${OUT} ====="
    conda run --no-capture-output -n "${CONDA_ENV}" \
        python profiler/measure_metrics.py \
        --config-name "experiment/${CFG}" \
        --n-passes "$N_PASSES" \
        --n-warmup-passes "$N_WARMUP" \
        --out "$OUT"
done

echo
echo "Profiles written to ${OUT_DIR}/"
