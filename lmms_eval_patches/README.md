# lmms-eval patches for release reproduction

PyPI `lmms_eval==0.7.1` lacks two custom behaviors required to reproduce
the paper's RefCOCO numbers across model families:

1. **`BBOX_COORD_FORMAT` env var** — controls how raw model output coords
   are de-normalized:
   - `normalized` — LLaVA-1.5 (already in `[0,1]`)
   - `pixel`      — Qwen2.5-VL (smart-resize pixels)
   - `scale_1000` — Qwen3-VL, Qwen3.5 (0-1000 scale)

2. **`REFCOCO_PROMPT_STYLE` env var** — controls user instruction:
   - `full`  — lmms-eval canonical Shikra / KOSMOS-2 prompt
   - `short` — trimmed
   - `nuwa`  — Nuwa-paper Qwen prompt: `Locate X ... JSON format.`

Three task files need patching (refcoco / refcoco+ / refcocog all use the
same env-var protocol):

```
lmms_eval/tasks/refcoco/utils_rec.py
lmms_eval/tasks/refcoco+/utils_rec.py
lmms_eval/tasks/refcocog/utils_rec.py
```

## Apply

```bash
bash apply.sh    # picks up the installed lmms_eval site-packages path
```

The script is idempotent — re-running after patch is applied is a no-op
(uses `patch --forward --quiet` which skips already-applied hunks).
