from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


def tensor_to_float_dict(values: Mapping[str, torch.Tensor | float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().cpu().item())
        else:
            result[key] = float(value)
    return result


def contribution_breakdown(
    mean_weights: Iterable[torch.Tensor],
    source_indices: Iterable[list[int]],
) -> dict[str, float]:
    embedding_values: list[float] = []
    early_values: list[float] = []
    late_values: list[float] = []
    entropy_values: list[float] = []

    for weights, indices in zip(mean_weights, source_indices):
        weights_cpu = weights.detach().cpu()
        if not indices:
            continue
        if 0 in indices:
            embedding_values.append(float(weights_cpu[indices.index(0)].item()))
        non_embedding_positions = [pos for pos, idx in enumerate(indices) if idx != 0]
        if non_embedding_positions:
            split = max(1, len(non_embedding_positions) // 2)
            early_pos = non_embedding_positions[:split]
            late_pos = non_embedding_positions[split:]
            early_values.append(float(weights_cpu[early_pos].sum().item()))
            late_values.append(float(weights_cpu[late_pos].sum().item()) if late_pos else 0.0)
        entropy = -(weights_cpu * (weights_cpu.clamp_min(1e-8).log())).sum().item()
        entropy_values.append(float(entropy))

    def _mean(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    return {
        "embedding_contribution": _mean(embedding_values),
        "early_contribution": _mean(early_values),
        "late_contribution": _mean(late_values),
        "depth_attention_entropy": _mean(entropy_values),
    }


def pad_depth_attention_rows(rows: list[torch.Tensor], fill_value: float = 0.0) -> torch.Tensor:
    if not rows:
        return torch.empty(0)
    width = max(row.numel() for row in rows)
    padded = []
    for row in rows:
        if row.numel() == width:
            padded.append(row)
            continue
        buffer = torch.full((width,), fill_value, dtype=row.dtype)
        buffer[: row.numel()] = row
        padded.append(buffer)
    return torch.stack(padded, dim=0)


def average_dicts(payloads: list[Mapping[str, float]]) -> dict[str, float]:
    if not payloads:
        return {}
    keys = sorted(payloads[0].keys())
    return {key: sum(payload[key] for payload in payloads) / len(payloads) for key in keys}


def latest_or_default(values: list[Any], default: Any) -> Any:
    return values[-1] if values else default
