"""Model patching for visual token routing.

Patches HuggingFace VLM decoder layers to inject token routing before
attention. Supports LLaVA-1.5 (LLaMA backbone), Qwen2.5-VL, and Qwen3-VL.

Action semantics:
- delete: mask-based surrogate for token removal. Column masking blocks
  attention to unselected tokens as K,V targets, but unselected tokens
  still keep active Q rows and therefore can attend to selected/text
  tokens. Their final layer output keeps the normal masked-attention +
  FFN result. Decode-time KV masking prevents new tokens from attending
  to "deleted" entries in the cache.
- physical_delete: real physical token removal. Unselected visual tokens
  are removed from the sequence at routing layers. Subsequent layers see
  a shorter sequence. KV cache only contains remaining tokens. Position
  IDs are preserved (GAP-style: original RoPE positions with gaps).
- residual_skip: mask-based routing. Column masking blocks attention to
  unselected tokens as K,V targets, but unselected tokens still compute
  masked attention internally before their layer output is overwritten
  with the input (residual bypass — full layer skip). Tokens can
  re-enter at later routing layers. Decode-time KV masking blocks stale
  cache entries.
- attn_skip: mask-based attention-only skip. Column masking again only
  blocks K,V access to unselected tokens. The full masked-attention pass
  still runs, but for unselected tokens its output is discarded and
  replaced with FFN-only updates from the original input:
  x + MLP(post_attention_layernorm(x)). Decode-time KV masking blocks
  stale attention cache entries.
- full_kv: mask-based, diagnostic only. Unselected tokens don't attend
  but ARE attended to as K,V targets. Requires bfloat16.
- compact_route: gather selected+text → compact K×K attention → FFN →
  scatter back to full sequence. Unselected tokens take residual (input
  passthrough). Combines physical_delete efficiency with recoverability.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from .router import TokenRouter, RoutingDecision
from .dispatcher import TokenDispatcher, DispatchInfo


@dataclass
class RoutingContext:
    """Shared state across decoder layers during a single forward pass.

    Attributes:
        visual_token_range: (start, end) absolute positions of visual
            tokens in the current sequence. Must be set before the first
            routing layer.
        router: the scoring strategy.
        dispatcher: the dispatch strategy (kept for API compat).
        action: "delete" | "residual_skip" | "full_kv".
        attn_weights_cache: attention weights stored from a prior layer,
            used by the router for scoring.
        routing_log: per-layer routing decisions, for analysis.
    """
    visual_token_range: tuple[int, int]
    router: TokenRouter
    dispatcher: TokenDispatcher
    action: str
    attn_weights_cache: Optional[Tensor] = None
    routing_log: dict[int, RoutingDecision] = field(default_factory=dict)
    # Layer module references (used by some helpers to detect attn type).
    _layer_modules: dict = field(default_factory=dict)
    # Physical-delete state propagated to downstream-layer decode-step
    # intercept. After physical_delete shrinks the seq at the decision
    # layer, downstream layers receive the compacted hidden_states but
    # the top-level model still passes the original full-L PE/pos_ids;
    # _compact_position_{embeddings,ids} cache the compacted versions.
    _compacted: bool = False
    _compact_position_embeddings: Optional[tuple] = None
    _compact_position_ids: Optional[Tensor] = None
    # Model family: "llava" (default) or "qwen25vl" / "qwen3vl" (for
    # future extension — Qwen hooks are kept end-to-end).
    _model_family: str = "llava"

    # Stagewise reroute state (compact_route_stagewise action).
    # Inside a "stage" (between two routing decisions) the sequence
    # stays compact across all layers; only at stage boundaries do we
    # scatter back to full L. Avoids the per-layer gather/scatter that
    # vanilla compact_route does.
    #
    # We snapshot ONLY the deferred (L-K) positions, not full L —
    # the K kept positions get overwritten by the in-stage compact
    # output at the next decision layer, so caching them is wasted
    # memory. Position embeddings / IDs are re-read from the outer
    # model on each forward (no cache needed).
    _stage_compact_kept_indices: Optional[Tensor] = None
    _stage_deferred_indices: Optional[Tensor] = None
    _stage_deferred_hidden: Optional[Tensor] = None
    _stage_full_seq_len: int = 0


def _get_scoring_layers(router: TokenRouter) -> list[int]:
    """Determine which layers need to output attention weights for scoring."""
    from .router import FastVRouter, PDropRouter
    if isinstance(router, FastVRouter):
        return [router.scoring_layer]
    elif isinstance(router, PDropRouter):
        # Capture attention at each drop layer for re-scoring
        return list(router.drop_layers)
    return [2]  # default


def _make_patched_forward(original_forward, layer_idx: int, ctx: RoutingContext):
    """Create a patched forward function for a decoder layer."""

    @functools.wraps(original_forward)
    def patched_forward(
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings=None,
        **kwargs,
    ) -> Tensor:
        seq_len = hidden_states.shape[1]

        # --- Stagewise compact intercept (release-only) ---
        # If we're inside an active stage (compact kept-set carried from
        # upstream routing layer), short-circuit BEFORE the is_prefill
        # check (which uses seq_len >= vis_end and fails for K << L).
        # Two sub-cases:
        #   - non-decision layer in stage: stay compact via _forward_compact_in_stage
        #   - decision layer (PDrop L=7/15/23 etc): scatter back to full L,
        #     re-route, gather new compact via _forward_compact_route_stagewise.
        if (
            ctx.action == "compact_route_stagewise"
            and ctx._stage_compact_kept_indices is not None
            and seq_len == ctx._stage_compact_kept_indices.shape[0]
        ):
            if ctx.router.is_decision_layer(layer_idx):
                return _forward_compact_route_stagewise(
                    original_forward, layer_idx, ctx,
                    hidden_states, attention_mask, position_ids,
                    past_key_values, use_cache, position_embeddings,
                    **kwargs,
                )
            return _forward_compact_in_stage(
                original_forward, layer_idx, ctx,
                hidden_states, attention_mask, position_ids,
                past_key_values, use_cache, position_embeddings,
                **kwargs,
            )

        # --- Physical compaction intercept ---
        # After physical delete at a prior layer, hidden_states is shorter
        # than the attention_mask/position_embeddings from the top-level model.
        # Use the compacted versions stored in ctx.
        if ctx._compacted and seq_len > 1 and position_embeddings is not None:
            # cos shape: (batch, seq, dim) for LLaMA, (3, batch, seq, dim) for Qwen MROPE
            cos = position_embeddings[0]
            pe_seq = cos.shape[2] if cos.ndim == 4 else cos.shape[1]
            if seq_len != pe_seq:
                position_embeddings = ctx._compact_position_embeddings
                position_ids = ctx._compact_position_ids
                attention_mask = None  # sdpa handles causal for compacted seq

        vis_start, vis_end = ctx.visual_token_range

        # Skip routing during autoregressive decode steps (seq_len=1)
        # or when the current sequence doesn't span the visual range.
        is_prefill = seq_len > 1 and seq_len >= vis_end and vis_end > vis_start

        if not is_prefill:
            # During decode at layers after physical delete: the top-level
            # model's causal mask is sized for layer 0's full KV cache,
            # but compacted layers have shorter KV. Drop the mask and let
            # sdpa handle causal masking with the correct KV length.
            if ctx._compacted and ctx.router.should_route(layer_idx):
                attention_mask = None

            return original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        scoring_layers = _get_scoring_layers(ctx.router)

        # --- Case 1: Scoring-only layer (e.g. FastV layer 2) ---
        if layer_idx in scoring_layers and not ctx.router.should_route(layer_idx):
            return _forward_scoring_layer(
                original_forward, layer_idx, ctx,
                hidden_states, attention_mask, position_ids,
                past_key_values, use_cache, position_embeddings, **kwargs,
            )

        # --- Case 2: Routing layer ---
        if ctx.router.should_route(layer_idx):
            if layer_idx in scoring_layers:
                try:
                    _capture_attention_weights(layer_idx, ctx, hidden_states,
                                               attention_mask, position_embeddings)
                except torch.cuda.OutOfMemoryError:
                    import warnings
                    seq_len_val = hidden_states.shape[1]
                    warnings.warn(
                        f"[OOM] _capture_attention_weights skipped at layer "
                        f"{layer_idx} (seq_len={seq_len_val}). "
                        f"Sample will run without routing.",
                        stacklevel=2,
                    )
                    torch.cuda.empty_cache()
                    ctx.attn_weights_cache = None
                    ctx._oom_skipped = True

            if ctx.attn_weights_cache is not None:
                if ctx.action == "physical_delete":
                    # Physical delete fires at the decision layer; later
                    # layers see the already-shrunk sequence and fall
                    # through to the vanilla forward.
                    if ctx.router.is_decision_layer(layer_idx):
                        return _forward_physical_delete(
                            original_forward, layer_idx, ctx,
                            hidden_states, attention_mask, position_ids,
                            past_key_values, use_cache, position_embeddings,
                            **kwargs,
                        )
                elif ctx.action == "compact_route":
                    # Per-layer compact gather + KÃK attention + scatter
                    # back to full L. Inner k_proj/v_proj run on K tokens,
                    # so KV cache writes K K,V (matches the original FastV/PDrop semantics for KV cache writes).
                    return _forward_compact_route(
                        original_forward, layer_idx, ctx,
                        hidden_states, attention_mask, position_ids,
                        past_key_values, use_cache, position_embeddings,
                        **kwargs,
                    )
                elif ctx.action == "compact_route_stagewise":
                    # Routing-layer entry; in-stage non-routing layers
                    # are caught by the early intercept at top of
                    # patched_forward (above).
                    if ctx.router.is_decision_layer(layer_idx):
                        return _forward_compact_route_stagewise(
                            original_forward, layer_idx, ctx,
                            hidden_states, attention_mask, position_ids,
                            past_key_values, use_cache, position_embeddings,
                            **kwargs,
                        )
                else:
                    raise ValueError(
                        f"Unsupported routing action: {ctx.action!r}. "
                        f"Release supports: physical_delete, compact_route, "
                        f"compact_route_stagewise."
                    )


        # --- Case 3: Normal layer ---
        # Stagewise reroute: when we're inside an active stage (compact
        # tokens carried over from upstream routing layer), keep this
        # non-routing layer compact too — gather PE / pos_ids and run
        # original_forward on the compact tensor without scattering back.
        # Decode steps (hidden_states.shape[1] == 1) take vanilla path.
        if (
            ctx.action == "compact_route_stagewise"
            and ctx._stage_compact_kept_indices is not None
            and hidden_states.shape[1] == ctx._stage_compact_kept_indices.shape[0]
        ):
            return _forward_compact_in_stage(
                original_forward, layer_idx, ctx,
                hidden_states, attention_mask, position_ids,
                past_key_values, use_cache, position_embeddings,
                **kwargs,
            )

        return original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    return patched_forward


def _capture_attention_weights(
    layer_idx: int, ctx: RoutingContext,
    hidden_states: Tensor, attention_mask: Tensor | None,
    position_embeddings,
) -> None:
    """Compute and cache attention weights from Q,K projection hooks.

    Runs the Q and K projections (plus layernorm) without running the
    full layer. Works with any attention backend by directly computing
    Q*K in fp32.
    """
    layer_module = ctx._layer_modules[layer_idx]
    attn = layer_module.self_attn
    head_dim = attn.head_dim
    batch, seq_len = hidden_states.shape[:2]

    # Compute Q, K from the layer's input (through layernorm + projection)
    normed = layer_module.input_layernorm(hidden_states)
    q_raw = attn.q_proj(normed)
    k = attn.k_proj(normed)

    # Some Qwen variants use a gated q_proj: output is (H * head_dim * 2), chunked
    # into (query, gate). We only need query for attention scoring;
    # gate is applied after attn_output in the real forward path.
    if ctx._model_family == "qwen35":
        q_raw = q_raw.view(batch, seq_len, -1, head_dim * 2)
        q, _gate = torch.chunk(q_raw, 2, dim=-1)  # (B, S, H, D)
    else:
        q = q_raw.view(batch, seq_len, -1, head_dim)
    k = k.view(batch, seq_len, -1, head_dim)

    # Apply QK normalization if present (Qwen3-VL)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    # Apply RoPE (LLaMA standard or Qwen MROPE)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if ctx._model_family == "qwen25vl":
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                apply_multimodal_rotary_pos_emb,
            )
            mrope_section = attn.config.rope_parameters["mrope_section"]
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        else:
            # LLaVA (standard RoPE) and Qwen3-VL (interleaved MROPE
            # already baked into cos/sin by rotary_emb.forward())
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    # Expand KV heads for GQA models
    num_kv_groups = attn.num_key_value_groups
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)

    # Compute attention weights in fp32
    scaling = attn.scaling
    attn_weights = torch.matmul(q.float(), k.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1)

    ctx.attn_weights_cache = attn_weights.detach()


def _forward_scoring_layer(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """Run a scoring-only layer: capture attention weights, then run normally.

    Used for FastV where the scoring layer is separate from the decision
    layer (e.g., scoring_layer=2, decision_layer=3).
    """
    _capture_attention_weights(
        layer_idx, ctx, hidden_states, attention_mask, position_embeddings,
    )

    return original_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )


def _apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to Q and K tensors.

    Compatible with LLaMA / Qwen2.5-VL / Qwen3-VL (full rotary) and
    Variants with partial rotary (rotary_dim = head_dim * partial_rotary_factor,
    trailing pass-through dims). Detected by comparing cos last-dim to q
    last-dim — if shorter, only rotate the leading slice.

    cos, sin: (batch, seq_len, rotary_dim) from the model's rotary_emb.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    if rotary_dim == head_dim:
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed

    # Partial rotary: rotate only the first rotary_dim, pass rest through.
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _forward_physical_delete(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """Apply routing via physical token removal.

    Physically removes unselected visual tokens from the sequence.
    After this layer, hidden_states is shorter. Subsequent layers
    receive the compacted sequence via ctx's stored position embeddings.
    KV cache naturally only contains remaining tokens — no decode-time
    masking needed.

    Position IDs are preserved (GAP-style: original RoPE positions with
    gaps where tokens were removed).
    """
    vis_start, vis_end = ctx.visual_token_range
    batch = hidden_states.shape[0]
    device = hidden_states.device
    seq_len = hidden_states.shape[1]

    # 1. Compute routing decision
    decision = ctx.router.compute_scores(
        layer_idx=layer_idx,
        attn_weights=ctx.attn_weights_cache,
        hidden_states=hidden_states,
        visual_token_range=ctx.visual_token_range,
    )
    ctx.routing_log[layer_idx] = decision

    # 2. Build full-sequence keep indices
    # Keep all text tokens + selected visual tokens
    selected_mask = decision.selected_mask  # (batch, num_visual)
    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    unsel_relative = (~selected_mask[0]).nonzero(as_tuple=True)[0]
    unsel_abs = unsel_relative + vis_start
    keep_mask[unsel_abs] = False
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]  # sorted ascending
    num_kept_visual = int(selected_mask[0].sum().item())

    # 3. Physically compact hidden_states
    hidden_states = hidden_states[:, kept_indices].contiguous()

    # 4. GAP position strategy: preserve original position IDs with gaps
    #    so kept tokens retain their original RoPE positions (paper §3
    #    GAP mechanism). Release supports only GAP.
    if position_ids is not None:
        position_ids = position_ids[:, kept_indices].contiguous()
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 4:
            # Qwen MROPE: (3, batch, seq, dim).
            cos = cos[:, :, kept_indices].contiguous()
            sin = sin[:, :, kept_indices].contiguous()
        else:
            # LLaMA / standard rotary: (batch, seq, dim).
            cos = cos[:, kept_indices].contiguous()
            sin = sin[:, kept_indices].contiguous()
        position_embeddings = (cos, sin)

    # 6. Drop the attention mask — sdpa handles causal masking for the
    # compacted sequence (tokens remain in original causal order).
    attention_mask = None

    # 7. Run the original layer forward with compacted inputs
    output = original_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    # 8. Update context for subsequent layers: post-delete decode-step
    #    intercept (line ~190) uses these to feed the compacted PE/pos_ids
    #    when the top-level model's outer cache mask is sized for full L.
    ctx._compacted = True
    ctx._compact_position_embeddings = position_embeddings
    ctx._compact_position_ids = position_ids
    ctx.visual_token_range = (vis_start, vis_start + num_kept_visual)

    return output


def _forward_compact_route(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """Apply routing via compact gather → K×K attention → scatter back.

    This is the core compact dispatch path:
      1. Gather selected + text tokens into compact sequence (length M+K)
      2. Run K×K attention on compact sequence (efficiency gain)
      3. Run FFN on compact sequence
      4. Scatter results back to full sequence
      5. Unselected tokens keep their input (residual bypass)

    Unlike mask-based routing (L×L attention with column masking), this
    actually reduces attention FLOPs from O(L²) to O((M+K)²).

    Unlike physical_delete, the full sequence is preserved — unselected
    tokens can be re-selected at later routing layers.
    """
    vis_start, vis_end = ctx.visual_token_range
    batch = hidden_states.shape[0]
    device = hidden_states.device
    seq_len = hidden_states.shape[1]

    layer_module = ctx._layer_modules[layer_idx]

    # 1. Compute routing decision
    decision = ctx.router.compute_scores(
        layer_idx=layer_idx,
        attn_weights=ctx.attn_weights_cache,
        hidden_states=hidden_states,
        visual_token_range=ctx.visual_token_range,
    )
    ctx.routing_log[layer_idx] = decision

    # 2. Build compact indices: all text + selected visual tokens
    selected_mask = decision.selected_mask  # (batch, num_visual)
    unselected_mask = ~selected_mask

    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    unsel_relative = unselected_mask[0].nonzero(as_tuple=True)[0]
    unsel_abs = unsel_relative + vis_start
    keep_mask[unsel_abs] = False
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]  # sorted ascending

    # ─── P-MOD-EQUIVALENT KV-SAVE SEMANTICS ───────────────────────────
    # Gather the kept subset BEFORE calling the layer, so the layer's
    # internal k_proj/v_proj run on K tokens and past_key_values.update
    # writes only K K,V per routed layer. This is what makes the paper
    # §4.3 KV-cache savings real (matches the original residual-skip semantics).
    # ───────────────────────────────────────────────────────────────────

    # 3. Gather compact hidden + position_ids + position_embeddings.
    compact_hidden = hidden_states[:, kept_indices].contiguous()
    compact_pos_ids = None
    if position_ids is not None:
        compact_pos_ids = position_ids[:, kept_indices].contiguous()

    compact_pe = None
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 4:
            # Qwen MROPE: (3, batch, seq, dim) — kept for future extension.
            compact_pe = (cos[:, :, kept_indices].contiguous(),
                          sin[:, :, kept_indices].contiguous())
        else:
            # LLaMA / standard rotary: (batch, seq, dim)
            compact_pe = (cos[:, kept_indices].contiguous(),
                          sin[:, kept_indices].contiguous())

    # 4. Run the decoder layer on compact (B, K, d).
    #    The layer's internal self_attn does k_proj/v_proj(compact_hidden)
    #    → past_key_values.update writes K K,V (true KV save).
    #    Attention is K×K via sdpa's causal handling.
    compact_out = original_forward(
        compact_hidden,
        attention_mask=None,
        position_ids=compact_pos_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=compact_pe,
        **kwargs,
    )

    # 5. Scatter compact output back to full L; deferred = input (residual).
    if isinstance(compact_out, tuple):
        h_compact = compact_out[0]
        output = hidden_states.clone()
        output[:, kept_indices] = h_compact
        return (output,) + compact_out[1:]
    output = hidden_states.clone()
    output[:, kept_indices] = compact_out
    return output


def _find_visual_token_range_llava(
    model,
    input_ids: Tensor,
    pixel_values: Optional[Tensor] = None,
) -> tuple[int, int]:
    """Find visual token positions for LLaVA-1.5.

    In LLaVA-1.5 with HF transformers, image tokens are marked by a
    special token (typically image_token_index = 32000). After the model
    merges image features, the visual tokens replace these placeholders.

    For LLaVA-1.5 with 336px input and CLIP-ViT-L/14:
    - 576 visual tokens (24x24 patches)
    """
    if pixel_values is None:
        return (0, 0)

    image_token_id = model.config.image_token_index
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(image_positions) == 0:
        return (0, 0)

    vis_start = image_positions[0].item()
    num_patches = pixel_values.shape[-1] // model.config.vision_config.patch_size
    num_visual = num_patches * num_patches
    vis_end = vis_start + num_visual

    return (vis_start, vis_end)


def _find_visual_token_range_qwen25vl(
    model,
    input_ids: Tensor,
    **kwargs,
) -> tuple[int, int]:
    """Find visual token positions for Qwen2.5-VL.

    Qwen2.5-VL uses <|vision_start|> and <|vision_end|> special tokens
    to mark the visual region in the input sequence.
    """
    config = model.config
    vision_start_id = config.vision_start_token_id
    vision_end_id = config.vision_end_token_id

    ids = input_ids[0]
    start_positions = (ids == vision_start_id).nonzero(as_tuple=True)[0]
    end_positions = (ids == vision_end_id).nonzero(as_tuple=True)[0]

    if len(start_positions) == 0 or len(end_positions) == 0:
        return (0, 0)

    vis_start = start_positions[0].item() + 1
    vis_end = end_positions[0].item()

    return (vis_start, vis_end)


# =====================================================================
# Main patching API
# =====================================================================


def patch_model_for_routing(
    model,
    router: TokenRouter,
    dispatcher: TokenDispatcher,
    action: str,
    model_family: str = "llava",
) -> RoutingContext:
    """Patch a HuggingFace VLM for visual token routing.

    Args:
        model: HF model (LlavaForConditionalGeneration is the primary
            target; qwen2.5-vl / qwen3-vl hooks are kept for future
            extension but not exercised in the release configs).
        router: scoring strategy (FastVRouter or PDropRouter).
        dispatcher: dispatch strategy (kept for API compat).
        action: "physical_delete" | "compact_route" | "compact_route_stagewise".
        model_family: "llava" (default), "qwen25vl", or "qwen3vl".

    Returns:
        RoutingContext that holds shared state and routing logs.
        The context's visual_token_range is set automatically by the
        pre-forward hook on each prefill.
    """
    if model_family in ("llava", "qwen25vl", "qwen3vl"):
        layers = model.model.language_model.layers
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    ctx = RoutingContext(
        visual_token_range=(0, 0),
        router=router,
        dispatcher=dispatcher,
        action=action,
    )
    ctx._model_family = model_family
    ctx._layer_modules = {i: layer for i, layer in enumerate(layers)}

    for i, layer in enumerate(layers):
        original_forward = layer.forward
        layer.forward = _make_patched_forward(original_forward, i, ctx)

    # Register a pre-forward hook on the outer model to auto-detect
    # visual token positions from input_ids before each forward pass.
    # This makes routing work automatically during lmms-eval without
    # manually setting visual_token_range per sample.
    find_range = get_visual_token_finder(model_family)

    def _pre_forward_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is None:
            return

        # Detect visual token range
        vis_range = find_range(module, input_ids,
                               pixel_values=kwargs.get("pixel_values"))
        if vis_range[1] > vis_range[0]:
            # New prefill with visual tokens — unconditional reset.
            # (After physical_delete, ctx.visual_token_range is the
            # compacted range which could coincidentally match a new
            # sample's true range, causing stale router decisions.)
            ctx.visual_token_range = vis_range
            ctx.router.reset()
            ctx.attn_weights_cache = None
            ctx.routing_log.clear()
            ctx._compacted = False
            ctx._compact_position_embeddings = None
            ctx._compact_position_ids = None
            # Stagewise reroute: clear stage state for fresh prefill.
            ctx._stage_compact_kept_indices = None
            ctx._stage_deferred_indices = None
            ctx._stage_deferred_hidden = None
            ctx._stage_full_seq_len = 0

    handle = model.register_forward_pre_hook(
        _pre_forward_hook, with_kwargs=True,
    )
    model._routing_pre_hook_handle = handle

    return ctx


def unpatch_model(model, model_family: str = "llava") -> None:
    """Remove routing patches, restoring original forward methods."""
    if model_family in ("llava", "qwen25vl", "qwen3vl", "qwen35"):
        layers = model.model.language_model.layers
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    for layer in layers:
        if hasattr(layer.forward, '__wrapped__'):
            layer.forward = layer.forward.__wrapped__
        else:
            layer.forward = type(layer).forward.__get__(layer, type(layer))

    # Remove the pre-forward hook we registered (stored as _routing_pre_hook)
    if hasattr(model, '_routing_pre_hook_handle'):
        model._routing_pre_hook_handle.remove()
        del model._routing_pre_hook_handle


def get_visual_token_finder(model_family: str):
    """Return the appropriate visual token range finder for a model family."""
    if model_family == "llava":
        return _find_visual_token_range_llava
    elif model_family in ("qwen25vl", "qwen3vl", "qwen35"):
        # Qwen-family models share the vision_start/end token paradigm
        return _find_visual_token_range_qwen25vl
    else:
        raise ValueError(f"Unknown model family: {model_family}")


# ─────────────────────────────────────────────────────────────────────
# (extension hooks for future model families)
# ─────────────────────────────────────────────────────────────────────

def _forward_compact_route_stagewise(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """Routing-layer entry under stagewise: scatter prior stage back to
    full L if needed, compute new routing, gather compact, run layer,
    return compact (no scatter back — passes to next in-stage layer)."""
    # 1. If we're entering from a prior stage, reconstruct full L from
    #    (deferred snapshot) ∪ (compact updates carried in hidden_states).
    #    Only the deferred (L-K) positions were cached; the K kept rows
    #    are overwritten by the incoming compact hidden_states.
    entered_from_prior_stage = ctx._stage_compact_kept_indices is not None
    if entered_from_prior_stage:
        kept_prev = ctx._stage_compact_kept_indices       # (K,)
        deferred_prev = ctx._stage_deferred_indices       # (L-K,)
        B = hidden_states.shape[0]
        d = hidden_states.shape[-1]
        L = ctx._stage_full_seq_len
        full_hidden = hidden_states.new_empty((B, L, d))
        full_hidden[:, deferred_prev] = ctx._stage_deferred_hidden
        full_hidden[:, kept_prev]     = hidden_states
        hidden_states = full_hidden
        # position_embeddings / position_ids passed in from the outer
        # model are already full-L (rotary_emb is invariant across
        # layers) — no cache needed.

    # 1b. Re-capture attention weights at THIS layer on the full-L hidden.
    #
    #     When we entered via the stagewise early intercept (from a prior
    #     stage), Case 2's `_capture_attention_weights` was bypassed.
    #     PDrop's `compute_scores` reads `ctx.attn_weights_cache`, so
    #     without this re-capture the second+ decision layers (e.g. L=6,
    #     L=13, L=20 for drop_layers=[3,6,13,20]) would all score with
    #     stale L=3 attention and effectively collapse to FastV behavior.
    #
    #     For the first decision (kept_prev is None, e.g. L=3) Case 2 has
    #     already captured, so we skip to avoid double-work.
    if entered_from_prior_stage and layer_idx in _get_scoring_layers(ctx.router):
        try:
            _capture_attention_weights(
                layer_idx, ctx, hidden_states,
                None, position_embeddings,
            )
        except torch.cuda.OutOfMemoryError:
            import warnings
            warnings.warn(
                f"[OOM] stagewise re-capture skipped at decision layer "
                f"{layer_idx} (seq_len={hidden_states.shape[1]}). "
                f"Routing falls back to prior-layer scores.",
                stacklevel=2,
            )
            torch.cuda.empty_cache()

    # 2. Compute routing decision on full-L hidden_states.
    decision = ctx.router.compute_scores(
        layer_idx=layer_idx,
        attn_weights=ctx.attn_weights_cache,
        hidden_states=hidden_states,
        visual_token_range=ctx.visual_token_range,
    )
    ctx.routing_log[layer_idx] = decision

    vis_start, vis_end = ctx.visual_token_range
    seq_len = hidden_states.shape[1]
    device = hidden_states.device

    selected_mask = decision.selected_mask
    unselected_mask = ~selected_mask

    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    unsel_relative = unselected_mask[0].nonzero(as_tuple=True)[0]
    unsel_abs = unsel_relative + vis_start
    keep_mask[unsel_abs] = False
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]
    deferred_indices = (~keep_mask).nonzero(as_tuple=True)[0]

    # 3. Snapshot ONLY the deferred (L-K) positions for next stage —
    #    the K kept positions will be overwritten by the in-stage
    #    compact output at the next decision layer.
    ctx._stage_compact_kept_indices = kept_indices
    ctx._stage_deferred_indices = deferred_indices
    ctx._stage_deferred_hidden = hidden_states[:, deferred_indices].clone()
    ctx._stage_full_seq_len = seq_len

    # 4. Gather compact + run layer + return compact.
    compact_hidden = hidden_states[:, kept_indices].contiguous()
    compact_pos_ids = (
        position_ids[:, kept_indices].contiguous()
        if position_ids is not None else None
    )
    compact_pe = None
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 4:
            compact_pe = (cos[:, :, kept_indices].contiguous(),
                          sin[:, :, kept_indices].contiguous())
        else:
            compact_pe = (cos[:, kept_indices].contiguous(),
                          sin[:, kept_indices].contiguous())

    return original_forward(
        compact_hidden,
        attention_mask=None,
        position_ids=compact_pos_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=compact_pe,
        **kwargs,
    )


def _forward_compact_in_stage(
    original_forward, layer_idx: int, ctx: RoutingContext,
    hidden_states, attention_mask, position_ids,
    past_key_values, use_cache, position_embeddings, **kwargs,
):
    """In-stage non-routing layer under stagewise: input is compact
    (B, K, d) carried over from upstream routing layer. Gather PE /
    pos_ids to compact and run the layer on compact, return compact."""
    kept_indices = ctx._stage_compact_kept_indices

    compact_pos_ids = (
        position_ids[:, kept_indices].contiguous()
        if position_ids is not None else None
    )
    compact_pe = None
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 4:
            compact_pe = (cos[:, :, kept_indices].contiguous(),
                          sin[:, :, kept_indices].contiguous())
        else:
            compact_pe = (cos[:, kept_indices].contiguous(),
                          sin[:, kept_indices].contiguous())

    return original_forward(
        hidden_states,
        attention_mask=None,
        position_ids=compact_pos_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=compact_pe,
        **kwargs,
    )
