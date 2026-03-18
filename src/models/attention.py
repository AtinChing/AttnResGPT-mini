from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        bias: bool = True,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if return_attention or not hasattr(F, "scaled_dot_product_attention"):
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = self.causal_mask[:seq_len, :seq_len]
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            output = torch.matmul(attn, v)
            attn_summary = attn.mean(dim=1)
        else:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            attn_summary = None

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.resid_dropout(self.out_proj(output))
        return output, attn_summary


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, *, bias: bool = True) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.activation = nn.GELU(approximate="tanh")
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return self.dropout(x)
