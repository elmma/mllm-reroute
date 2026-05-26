#!/bin/bash
# Overlay 3 refcoco utils_rec.py patches on top of an editable install of
# lmms-eval v0.7.1. Adds two env vars (BBOX_COORD_FORMAT,
# REFCOCO_PROMPT_STYLE) plus the matching RefCOCO prompt branch needed
# for Qwen2.5-VL reproduction. Idempotent — safe to re-run.
#
# Usage:
#   bash apply.sh                 # auto-detect lmms_eval install path
#   bash apply.sh <path/to/site>  # override (advanced)

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate installed lmms_eval
if [[ "${1:-}" ]]; then
    LMMS_EVAL_ROOT="$1"
else
    LMMS_EVAL_ROOT="$(python -c 'import lmms_eval, os; print(os.path.dirname(lmms_eval.__file__))')"
fi

if [[ ! -d "$LMMS_EVAL_ROOT" ]]; then
    echo "ERROR: lmms_eval not found at $LMMS_EVAL_ROOT" >&2
    echo "Install upstream v0.7.1 first:" >&2
    echo "  git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git" >&2
    echo "  pip install -e ./lmms-eval" >&2
    exit 1
fi

echo "Overlaying refcoco patches on: $LMMS_EVAL_ROOT"

for task in refcoco refcoco+ refcocog; do
    target="$LMMS_EVAL_ROOT/tasks/${task}/utils_rec.py"
    patch="$PATCH_DIR/${task}_utils_rec.patch"
    if [[ ! -f "$target" ]]; then
        echo "  SKIP $task — target missing ($target)" >&2
        continue
    fi
    if [[ ! -f "$patch" ]]; then
        echo "  SKIP $task — patch missing ($patch)" >&2
        continue
    fi
    # Idempotency-first: check if already patched BEFORE running patch
    # (avoids the noisy "Reversed (or previously applied)" + .rej spam).
    if grep -q "BBOX_COORD_FORMAT" "$target" && grep -q "REFCOCO_PROMPT_STYLE" "$target"; then
        echo "  ✓ lmms_eval/tasks/${task}/utils_rec.py already patched (skip)"
        continue
    fi
    SITE_PKGS="$(dirname "$LMMS_EVAL_ROOT")"
    if patch --forward --quiet -p0 -d "$SITE_PKGS" < "$patch" 2>/dev/null; then
        echo "  ✓ patched lmms_eval/tasks/${task}/utils_rec.py"
    else
        rm -f "${target}.rej"
        echo "  ✗ FAILED to patch ${task} — manual review needed" >&2
        exit 1
    fi
done

echo
echo "Patches applied. Verify:"
echo "  python -c 'from lmms_eval.tasks import TaskManager; TaskManager(); print(\"OK\")'"
echo "  python -c 'from lmms_eval.tasks.refcoco.utils_rec import BBOX_COORD_FORMAT, REFCOCO_PROMPT_STYLE; print(BBOX_COORD_FORMAT, REFCOCO_PROMPT_STYLE)'"
