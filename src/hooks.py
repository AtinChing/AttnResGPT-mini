from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn

from src.models.attention import CausalSelfAttention, FeedForward


@dataclass
class NormHookCollector:
    activation_norms: Dict[str, float] = field(default_factory=dict)
    gradient_norms: Dict[str, float] = field(default_factory=dict)
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def _forward_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            value = output.detach().float().norm(dim=-1).mean().item()
            self.activation_norms[name] = float(value)

        return hook

    def _backward_hook(self, name: str):
        def hook(
            _module: nn.Module,
            _grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor | None, ...],
        ) -> None:
            grad = grad_output[0]
            if grad is None:
                return
            value = grad.detach().float().norm(dim=-1).mean().item()
            self.gradient_norms[name] = float(value)

        return hook

    def register(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, (CausalSelfAttention, FeedForward)):
                self.handles.append(module.register_forward_hook(self._forward_hook(name)))
                self.handles.append(module.register_full_backward_hook(self._backward_hook(name)))

    def reset_step(self) -> None:
        self.activation_norms.clear()
        self.gradient_norms.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
