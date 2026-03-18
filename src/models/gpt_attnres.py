from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.metrics import contribution_breakdown
from src.models.attention import RMSNorm
from src.models.attnres import DepthAttentionResidual
from src.models.blocks import AttnResTransformerBlock


class GPTAttnRes(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [AttnResTransformerBlock(config, layer_index=i) for i in range(config.n_layers)]
        )
        self.final_residual = DepthAttentionResidual(
            config.d_model,
            temperature=config.attnres.temperature,
            window_size=config.attnres.window_size,
            rmsnorm_keys=config.attnres.rmsnorm_keys,
            zero_init_query=config.attnres.zero_init_queries,
            include_embedding=config.attnres.include_embedding,
            keep_embedding_in_window=config.attnres.keep_embedding_in_window,
            eps=config.norm_eps,
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
        x0 = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x0 = self.dropout(x0)
        history: list[torch.Tensor] = [x0]

        depth_attention_rows: list[torch.Tensor] = []
        depth_source_indices: list[list[int]] = []
        block_output_norms: list[float] = []

        for block in self.blocks:
            history, block_aux = block(
                history,
                ablate_sublayers=ablate_sublayers,
                return_aux=return_aux,
            )
            if return_aux:
                block_output_norms.append(block_aux["block_output_norm"])
                for row in block_aux["depth_attention"]:
                    depth_attention_rows.append(row["mean_weights"])
                    depth_source_indices.append(row["source_indices"])

        if self.config.attnres.final_readout:
            x, final_stats = self.final_residual(history, return_stats=return_aux)
            if return_aux:
                depth_attention_rows.append(final_stats["mean_weights"])
                depth_source_indices.append(final_stats["source_indices"])
        else:
            x = history[-1]

        x = self.final_norm(x)
        logits = self.lm_head(x)

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "block_output_norms": block_output_norms,
                "depth_attention": depth_attention_rows,
                "source_indices": depth_source_indices,
                **contribution_breakdown(depth_attention_rows, depth_source_indices),
            }
        return logits, aux
