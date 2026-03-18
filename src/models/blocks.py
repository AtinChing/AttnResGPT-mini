from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.attention import CausalSelfAttention, FeedForward, RMSNorm
from src.models.attnres import DepthAttentionResidual


class BaselineTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attn_sublayer_index = 2 * layer_index
        self.mlp_sublayer_index = 2 * layer_index + 1
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            bias=config.bias,
            max_seq_len=config.max_seq_len,
        )
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ablate_sublayers: Optional[set[int]] = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        attn_out, _ = self.attn(self.attn_norm(x))
        if ablate_sublayers and self.attn_sublayer_index in ablate_sublayers:
            attn_out = torch.zeros_like(attn_out)
        x = x + attn_out

        mlp_out = self.mlp(self.mlp_norm(x))
        if ablate_sublayers and self.mlp_sublayer_index in ablate_sublayers:
            mlp_out = torch.zeros_like(mlp_out)
        x = x + mlp_out

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "block_output_norm": float(x.detach().float().norm(dim=-1).mean().item()),
            }
        return x, aux


class AttnResTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attn_sublayer_index = 2 * layer_index
        self.mlp_sublayer_index = 2 * layer_index + 1
        attnres = config.attnres
        self.pre_attn_res = DepthAttentionResidual(
            config.d_model,
            temperature=attnres.temperature,
            window_size=attnres.window_size,
            rmsnorm_keys=attnres.rmsnorm_keys,
            zero_init_query=attnres.zero_init_queries,
            include_embedding=attnres.include_embedding,
            keep_embedding_in_window=attnres.keep_embedding_in_window,
            eps=config.norm_eps,
        )
        self.pre_mlp_res = DepthAttentionResidual(
            config.d_model,
            temperature=attnres.temperature,
            window_size=attnres.window_size,
            rmsnorm_keys=attnres.rmsnorm_keys,
            zero_init_query=attnres.zero_init_queries,
            include_embedding=attnres.include_embedding,
            keep_embedding_in_window=attnres.keep_embedding_in_window,
            eps=config.norm_eps,
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            bias=config.bias,
            max_seq_len=config.max_seq_len,
        )
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout, bias=config.bias)

    def forward(
        self,
        history: list[torch.Tensor],
        *,
        ablate_sublayers: Optional[set[int]] = None,
        return_aux: bool = False,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        attn_input, attn_res_stats = self.pre_attn_res(history, return_stats=return_aux)
        attn_out, _ = self.attn(self.attn_norm(attn_input))
        if ablate_sublayers and self.attn_sublayer_index in ablate_sublayers:
            attn_out = torch.zeros_like(attn_out)
        history.append(attn_out)

        mlp_input, mlp_res_stats = self.pre_mlp_res(history, return_stats=return_aux)
        mlp_out = self.mlp(self.mlp_norm(mlp_input))
        if ablate_sublayers and self.mlp_sublayer_index in ablate_sublayers:
            mlp_out = torch.zeros_like(mlp_out)
        history.append(mlp_out)

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention": [
                    {"name": f"block_{self.layer_index:02d}_pre_attn", **attn_res_stats},
                    {"name": f"block_{self.layer_index:02d}_pre_mlp", **mlp_res_stats},
                ],
                "block_output_norm": float(mlp_out.detach().float().norm(dim=-1).mean().item()),
            }
        return history, aux
