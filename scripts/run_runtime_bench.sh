#!/bin/bash
# Runtime benchmark dispatcher — loops release configs and writes one
# JSON per config into experiments/runtime/.
#
# Usage:
#   bash scripts/run_runtime_bench.sh <model> [--tier <T>] \
#        [--n-passes N] [--n-warmup-passes N] [--n-decode-tokens N] [--gpu <id>]
#
# model:  llava15 | qwen25vl | all
# --tier: avg192 (default) | avg128 | avg64 | all
#
# Per (model, tier): 1 baseline + 6 routing variants =
#   fastv_K3, pdrop_earlyL2, ours_vs_fastv, ours_vs_pdrop,
#   ours_vs_fastv_stagewise, ours_vs_pdrop_stagewise

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [--tier avg192|avg128|avg64|all] [--n-passes N] [--n-warmup-passes N] [--n-decode-tokens N] [--gpu <id>]}"
shift

TIER="avg192"
N_PASSES=5
N_WARMUP=2
N_DECODE=64
GPU_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)              TIER="$2";    shift 2 ;;
        --tier=*)            TIER="${1#--tier=}"; shift ;;
        --n-passes)          N_PASSES="$2"; shift 2 ;;
        --n-warmup-passes)   N_WARMUP="$2"; shift 2 ;;
        --n-decode-tokens)   N_DECODE="$2"; shift 2 ;;
        --gpu)               GPU_ARG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$GPU_ARG" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ARG"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PROJECT_ROOT}/experiments/runtime"
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

echo "[runtime dispatcher] tier=$TIER  model=$MODEL  configs=${#CONFIGS[@]}  n_passes=$N_PASSES  n_warmup=$N_WARMUP  n_decode=$N_DECODE"

for CFG in "${CONFIGS[@]}"; do
    TAG="${CFG//\//_}"
    OUT="${OUT_DIR}/${TAG}.json"
    echo
    echo "===== $(date '+%F %T'): RUNTIME ${CFG} → ${OUT} ====="
    conda run --no-capture-output -n "${CONDA_ENV}" \
        python profiler/bench_runtime.py \
        --config-name "experiment/${CFG}" \
        --n-passes "$N_PASSES" \
        --n-warmup-passes "$N_WARMUP" \
        --n-decode-tokens "$N_DECODE" \
        --out "$OUT"
done

echo
echo "Runtime JSONs written to ${OUT_DIR}/"
