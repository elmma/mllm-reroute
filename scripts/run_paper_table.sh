#!/bin/bash
# Reproduce the paper main table — thin dispatcher that enumerates every
# release config and calls run_setting.sh per (config, task) pair.
#
# Usage:
#   bash scripts/run_paper_table.sh <model> [--tier <T>] \
#        [--tasks <sel>] [--limit <N>] [--gpu <id>]
#
# model:
#   llava15     LLaVA-1.5-7B configs for selected tier(s)
#   qwen25vl    Qwen2.5-VL-7B configs for selected tier(s)
#   all         both models
#
# --tier T:
#   avg192      (default — 5 routing configs per model)
#   avg128
#   avg64
#   all         all three tiers (15 routing configs per model)
#
# Per tier each model has the 6 routing variants:
#   fastv_K3                  FastV (single-shot, scoring at L3) vanilla physical-delete
#   pdrop_earlyL2             PDrop (multi-stage cascade from L2) vanilla physical-delete
#   ours_vs_fastv             Ours reroute, head-to-head FLOPs-matched to fastv_K3
#   ours_vs_pdrop             Ours reroute, head-to-head FLOPs-matched to pdrop_earlyL2
#   ours_vs_fastv_stagewise   Stagewise dispatch variant of ours_vs_fastv (runtime)
#   ours_vs_pdrop_stagewise   Stagewise dispatch variant of ours_vs_pdrop (runtime)
#
# Baselines (no routing) are included once per model irrespective of tier.
#
# Remaining flags forward to run_setting.sh.

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [--tier avg192|avg128|avg64|all] [--tasks <sel>] [--limit <N>] [--gpu <id>]}"
shift

TIER="avg192"
PASSTHROUGH=()
while (( $# > 0 )); do
    case "$1" in
        --tier) TIER="$2"; shift 2 ;;
        --tier=*) TIER="${1#--tier=}"; shift ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# Build config paths matching configs/experiment/<model>/<tier>/<model>_<variant>_<tier>.yaml
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

echo "[dispatcher] tier=$TIER  model=$MODEL  configs=${#CONFIGS[@]}"

for CFG in "${CONFIGS[@]}"; do
    bash "${PROJECT_ROOT}/scripts/run_setting.sh" "$CFG" "${PASSTHROUGH[@]}"
done
