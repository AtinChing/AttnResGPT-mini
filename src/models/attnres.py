from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.attention import RMSNorm


class DepthAttentionResidual(nn.Module):
    """Depth-wise softmax aggregation over previous layer outputs."""

    def __init__(
        self,
        d_model: int,
        *,
        temperature: float = 1.0,
        window_size: int | None = None,
        rmsnorm_keys: bool = True,
        zero_init_query: bool = True,
        include_embedding: bool = True,
        keep_embedding_in_window: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.window_size = window_size
        self.include_embedding = include_embedding
        self.keep_embedding_in_window = keep_embedding_in_window
        self.query = nn.Parameter(torch.empty(d_model))
        self.key_norm = RMSNorm(d_model, eps=eps) if rmsnorm_keys else nn.Identity()
        if zero_init_query:
            nn.init.zeros_(self.query)
        else:
            nn.init.normal_(self.query, mean=0.0, std=0.02)

    def _select_history(self, history: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[int]]:
        if not history:
            raise ValueError("history must contain at least one source tensor")
        indices = list(range(len(history)))
        if not self.include_embedding and indices:
            history = history[1:]
            indices = indices[1:]
        if self.window_size is None or len(history) <= self.window_size + int(self.keep_embedding_in_window):
            return history, indices

        if self.keep_embedding_in_window and indices:
            return [history[0], *history[-self.window_size :]], [indices[0], *indices[-self.window_size :]]
        return history[-self.window_size :], indices[-self.window_size :]

    def forward(
        self,
        history: list[torch.Tensor],
        *,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        selected_history, selected_indices = self._select_history(history)
        values = torch.stack(selected_history, dim=0)  # [S, B, T, D]
        keys = self.key_norm(values)
        logits = torch.einsum("d,sbtd->sbt", self.query, keys)
        logits = logits / max(self.temperature, 1e-6)
        weights = torch.softmax(logits, dim=0)
        mixed = torch.einsum("sbt,sbtd->btd", weights, values)

        stats: dict[str, Any] = {}
        if return_stats:
            mean_weights = weights.detach().mean(dim=(1, 2)).cpu()
            entropy = -(weights.detach() * weights.detach().clamp_min(1e-8).log()).sum(dim=0).mean().cpu()
            stats = {
                "source_indices": selected_indices,
                "mean_weights": mean_weights,
                "entropy": float(entropy.item()),
            }
        return mixed, stats
