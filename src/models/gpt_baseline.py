from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.attention import RMSNorm
from src.models.blocks import BaselineTransformerBlock


class GPTBaseline(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [BaselineTransformerBlock(config, layer_index=i) for i in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    @property
    def num_sublayers(self) -> int:
        return 2 * self.config.n_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        ablate_sublayers: Optional[set[int]] = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("Input sequence is longer than model.max_seq_len")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        block_norms: list[float] = []
        for block in self.blocks:
            x, block_aux = block(x, ablate_sublayers=ablate_sublayers, return_aux=return_aux)
            if return_aux:
                block_norms.append(block_aux["block_output_norm"])

        x = self.final_norm(x)
        logits = self.lm_head(x)
        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "block_output_norms": block_norms,
                "depth_attention": [],
                "source_indices": [],
            }
        return logits, aux
