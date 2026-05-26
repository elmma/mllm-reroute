"""Token dispatch strategies for visual token routing.

Implements three actions applied at each routing layer:

A. **delete** — permanently remove unselected visual tokens from the
   sequence (FastV-style). Sequence length shrinks.
B. **residual_skip** — selected tokens form a compact K-token sequence
   for attention (K×K). Unselected tokens bypass via residual. Sequence
   length stays L. Tokens can re-enter at later layers.
C. **full_kv** — all L tokens compute K,V projections. Only selected
   tokens compute Q. Attention is K×L. Unselected tokens take residual
   only. Diagnostic ablation: no compute saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class DispatchInfo:
    """Metadata needed to scatter attention output back to full sequence.

    Attributes:
        action: which dispatch strategy was used.
        full_length: original sequence length L (including text tokens).
        visual_token_range: (start, end) in the original sequence.
        selected_indices_abs: (batch, K) absolute positions in the
            original sequence for the selected visual tokens.
        original_hidden: (batch, L, d) hidden states before dispatch,
            needed for residual merge in residual_skip and full_kv.
        all_non_visual_indices: (batch, M) indices of system + text tokens
            that are always kept active.
    """
    action: str
    full_length: int
    visual_token_range: tuple[int, int]
    selected_indices_abs: Tensor
    original_hidden: Tensor
    all_non_visual_indices: Optional[Tensor] = None


class TokenDispatcher:
    """Manages token gather/scatter for routing at each decoder layer."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(
        self,
        hidden_states: Tensor,
        selected_indices: Tensor,
        visual_token_range: tuple[int, int],
        action: str,
    ) -> tuple[Tensor, Tensor, DispatchInfo]:
        """Prepare tokens for attention based on routing decision.

        Args:
            hidden_states: (batch, L, d) full sequence.
            selected_indices: (batch, K) indices into the visual range
                (0-based relative to visual_token_range start). Sorted.
            visual_token_range: (start, end) absolute positions.
            action: "delete" | "residual_skip" | "full_kv".

        Returns:
            q_input: tokens that compute Q projections.
            kv_input: tokens that appear as K,V in attention.
            info: DispatchInfo for combine() step.
        """
        vis_start, vis_end = visual_token_range
        batch, seq_len, dim = hidden_states.shape

        # Convert relative visual indices to absolute sequence positions
        selected_abs = selected_indices + vis_start  # (batch, K)

        # Non-visual tokens: everything outside [vis_start, vis_end)
        device = hidden_states.device
        all_indices = torch.arange(seq_len, device=device)
        non_visual_mask = (all_indices < vis_start) | (all_indices >= vis_end)
        non_visual_indices = all_indices[non_visual_mask]  # (M,)
        # Expand for batch: (batch, M)
        non_visual_indices = non_visual_indices.unsqueeze(0).expand(batch, -1)

        info = DispatchInfo(
            action=action,
            full_length=seq_len,
            visual_token_range=visual_token_range,
            selected_indices_abs=selected_abs,
            original_hidden=hidden_states,
            all_non_visual_indices=non_visual_indices,
        )

        if action == "delete":
            return self._dispatch_delete(
                hidden_states, selected_abs, non_visual_indices, info
            )
        elif action == "residual_skip":
            return self._dispatch_residual_skip(
                hidden_states, selected_abs, non_visual_indices, info
            )
        elif action == "full_kv":
            return self._dispatch_full_kv(
                hidden_states, selected_abs, non_visual_indices, info
            )
        else:
            raise ValueError(f"Unknown action: {action}")

    def combine(
        self,
        attn_output: Tensor,
        info: DispatchInfo,
    ) -> Tensor:
        """Scatter attention output back and merge with residual.

        Args:
            attn_output: output from attention layer, shape depends on
                action:
                - delete: (batch, M+K, d) — shortened sequence
                - residual_skip: (batch, M+K, d) — active tokens only
                - full_kv: (batch, M+K, d) — Q-active tokens only
            info: DispatchInfo from dispatch().

        Returns:
            (batch, L', d) where L' = M+K for delete, L for others.
        """
        if info.action == "delete":
            return self._combine_delete(attn_output, info)
        elif info.action == "residual_skip":
            return self._combine_residual_skip(attn_output, info)
        elif info.action == "full_kv":
            return self._combine_full_kv(attn_output, info)
        else:
            raise ValueError(f"Unknown action: {info.action}")

    # ------------------------------------------------------------------
    # Action A: Delete
    # ------------------------------------------------------------------

    def _dispatch_delete(
        self,
        hidden_states: Tensor,
        selected_abs: Tensor,
        non_visual_indices: Tensor,
        info: DispatchInfo,
    ) -> tuple[Tensor, Tensor, DispatchInfo]:
        """Permanently remove unselected visual tokens.

        The resulting sequence contains only non-visual tokens + selected
        visual tokens. Both Q and KV operate on this shortened sequence.
        """
        # Build the keep indices: non_visual + selected visual, sorted
        keep_indices = torch.cat([non_visual_indices, selected_abs], dim=1)
        keep_indices = keep_indices.sort(dim=1).values  # (batch, M+K)

        # Gather tokens — same for Q and KV
        compact = self._batch_gather(hidden_states, keep_indices)
        return compact, compact, info

    def _combine_delete(
        self, attn_output: Tensor, info: DispatchInfo
    ) -> Tensor:
        # For delete, the sequence is already shortened. Just pass through.
        return attn_output

    # ------------------------------------------------------------------
    # Action B: Residual Skip
    # ------------------------------------------------------------------

    def _dispatch_residual_skip(
        self,
        hidden_states: Tensor,
        selected_abs: Tensor,
        non_visual_indices: Tensor,
        info: DispatchInfo,
    ) -> tuple[Tensor, Tensor, DispatchInfo]:
        """Selected visual tokens + all non-visual tokens form the compact
        sequence for attention (K×K style). Unselected visual tokens will
        bypass via residual in combine().
        """
        # Active tokens: non_visual + selected visual
        active_indices = torch.cat([non_visual_indices, selected_abs], dim=1)
        active_indices = active_indices.sort(dim=1).values  # (batch, M+K)

        compact = self._batch_gather(hidden_states, active_indices)
        # Q and KV both operate on the compact set
        return compact, compact, info

    def _combine_residual_skip(
        self, attn_output: Tensor, info: DispatchInfo
    ) -> Tensor:
        """Scatter attention output to active positions, residual for rest."""
        batch, _, dim = attn_output.shape
        device = attn_output.device
        full_len = info.full_length

        # Start from the original hidden states (residual for all)
        output = info.original_hidden.clone()

        # Build active indices (same logic as dispatch)
        active_indices = torch.cat(
            [info.all_non_visual_indices, info.selected_indices_abs], dim=1
        )
        active_indices = active_indices.sort(dim=1).values  # (batch, M+K)

        # Scatter attention output into active positions
        self._batch_scatter(output, active_indices, attn_output)

        return output

    # ------------------------------------------------------------------
    # Action C: Full KV (diagnostic)
    # ------------------------------------------------------------------

    def _dispatch_full_kv(
        self,
        hidden_states: Tensor,
        selected_abs: Tensor,
        non_visual_indices: Tensor,
        info: DispatchInfo,
    ) -> tuple[Tensor, Tensor, DispatchInfo]:
        """Q computed only for active tokens (non-visual + selected visual).
        K,V computed for ALL tokens (full sequence).

        This means attention is (M+K) queries × L keys.
        """
        # Q input: compact (non-visual + selected visual)
        active_indices = torch.cat([non_visual_indices, selected_abs], dim=1)
        active_indices = active_indices.sort(dim=1).values
        q_input = self._batch_gather(hidden_states, active_indices)

        # KV input: full sequence
        kv_input = hidden_states

        return q_input, kv_input, info

    def _combine_full_kv(
        self, attn_output: Tensor, info: DispatchInfo
    ) -> Tensor:
        """Same as residual_skip: scatter active outputs, residual for rest."""
        # Reuse the residual_skip combine — same logic
        return self._combine_residual_skip(attn_output, info)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_gather(tensor: Tensor, indices: Tensor) -> Tensor:
        """Gather along seq dimension for each batch element.

        Args:
            tensor: (batch, seq_len, dim)
            indices: (batch, K) — positions to gather

        Returns:
            (batch, K, dim)
        """
        # Expand indices to match hidden dim
        expanded = indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
        return torch.gather(tensor, 1, expanded)

    @staticmethod
    def _batch_scatter(
        target: Tensor, indices: Tensor, source: Tensor
    ) -> None:
        """In-place scatter along seq dimension for each batch element.

        Args:
            target: (batch, seq_len, dim) — modified in-place.
            indices: (batch, K) — positions to write to.
            source: (batch, K, dim) — values to write.
        """
        expanded = indices.unsqueeze(-1).expand(-1, -1, target.shape[-1])
        target.scatter_(1, expanded, source)
