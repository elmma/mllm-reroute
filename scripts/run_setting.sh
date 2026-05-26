#!/bin/bash
# Run a single experiment config across one or more eval tasks.
#
# Usage:
#   bash scripts/run_setting.sh <config_path> \
#        [--tasks <sel>] [--limit <N>] [--gpu <id>] [--log-dir <base>]
#
# config_path (path under configs/experiment/, without .yaml):
#   baseline/llava15
#   baseline/qwen25vl
#   llava15/<tier>/llava15_<method>_<tier>
#   qwen25vl/<tier>/qwen25vl_<method>_<tier>
#
# where <tier> ∈ {avg192, avg128, avg64} and <method> ∈
#   {fastv_K3, pdrop_earlyL2, ours_vs_fastv, ours_vs_pdrop,
#    ours_vs_fastv_stagewise}
#
# Examples:
#   bash scripts/run_setting.sh baseline/llava15
#   bash scripts/run_setting.sh llava15/avg192/llava15_ours_vs_fastv_avg192
#   bash scripts/run_setting.sh qwen25vl/avg64/qwen25vl_ours_vs_pdrop_avg64
#
# --tasks selectors:
#   pope         POPE only                                              [default]
#   refcoco      refcoco_val refcoco_testA refcoco_testB
#   grounding    8 splits: refcoco{,+,g} × {val,testA,testB,test}
#   vqa          gqa mmbench mme (VQA-style benchmarks)
#   ablation     4-task minimal set: gqa mmbench refcoco_testA refcoco_testB
#   paper_main   single multi-bench eval — POPE + 8 grounding splits
#   <csv>        comma-separated custom, e.g. pope,refcoco_testA,mme
#
# --limit N:    pass +eval.limit=N (smoke-test mode), default unlimited.
# --gpu <id>:   sets CUDA_VISIBLE_DEVICES=<id>.
# --log-dir <base>: override base log directory.
#               Default: <project_root>/experiments/logs/<config_path>/.
#
# Env vars:
#   CONDA_ENV   conda env name (default: reroute_release).

set -euo pipefail

CONFIG="${1:?Usage: $0 <config_path> [--tasks <sel>] [--limit <N>] [--gpu <id>]}"
shift

TASKS_ARG="pope"
LIMIT_ARG=""
GPU_ARG=""
LOG_DIR_BASE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)   TASKS_ARG="$2"; shift 2 ;;
        --limit)   LIMIT_ARG="+eval.limit=$2"; shift 2 ;;
        --gpu)     GPU_ARG="$2"; shift 2 ;;
        --log-dir) LOG_DIR_BASE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$GPU_ARG" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ARG"
fi

# --- Task selectors ---
GROUNDING=("refcoco_val" "refcoco_testA" "refcoco_testB"
           "refcoco+_val" "refcoco+_testA" "refcoco+_testB"
           "refcocog_val" "refcocog_test")
REFCOCO=("refcoco_val" "refcoco_testA" "refcoco_testB")
VQA=("gqa" "mmbench" "mme")
# Fast iteration set: 1 general VQA + 1 multi-skill VQA + 2 grounding splits.
ABLATION=("gqa" "mmbench" "refcoco_testA" "refcoco_testB")

case "$TASKS_ARG" in
    pope)       TASKS=("pope") ;;
    refcoco)    TASKS=("${REFCOCO[@]}") ;;
    grounding)  TASKS=("${GROUNDING[@]}") ;;
    vqa)        TASKS=("${VQA[@]}") ;;
    ablation)   TASKS=("${ABLATION[@]}") ;;
    paper_main) TASKS=("paper_main") ;;
    *)          IFS=',' read -ra TASKS <<< "$TASKS_ARG" ;;
esac

CONDA_ENV="${CONDA_ENV:-reroute_release}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve log-dir base
if [[ -z "$LOG_DIR_BASE" ]]; then
    LOG_DIR_BASE="${PROJECT_ROOT}/experiments/logs"
elif [[ "$LOG_DIR_BASE" != /* ]]; then
    LOG_DIR_BASE="${PROJECT_ROOT}/${LOG_DIR_BASE}"
fi
LOG_DIR="${LOG_DIR_BASE}/${CONFIG}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Config:    ${CONFIG}"
echo "Tasks:     ${TASKS[*]}"
echo "Log dir:   ${LOG_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "GPU(s):    ${CUDA_VISIBLE_DEVICES:-<inherit/all>}"
[[ -n "$LIMIT_ARG" ]] && echo "Limit:     ${LIMIT_ARG}"
echo "================================================================"

cd "${PROJECT_ROOT}"

for TASK in "${TASKS[@]}"; do
    LOG_FILE="${LOG_DIR}/${TASK}.log"
    HYDRA_DIR="${LOG_DIR}/${TASK}"
    echo
    echo "===== $(date '+%F %T'): START ${CONFIG} x ${TASK} ====="
    if conda run --no-capture-output -n "${CONDA_ENV}" \
        python scripts/run_eval.py \
        --config-dir "${PROJECT_ROOT}/configs" \
        --config-name "experiment/${CONFIG}" \
        eval=${TASK} \
        ${LIMIT_ARG} \
        "hydra.run.dir=${HYDRA_DIR}" \
        2>&1 | tee "${LOG_FILE}"; then
        echo "===== $(date '+%F %T'): DONE  ${CONFIG} x ${TASK} ====="
    else
        echo "===== $(date '+%F %T'): FAIL  ${CONFIG} x ${TASK} (continuing) ====="
    fi
done

echo
echo "All tasks finished for ${CONFIG}. Logs: ${LOG_DIR}"
