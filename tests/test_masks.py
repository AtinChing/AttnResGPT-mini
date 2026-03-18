from __future__ import annotations

import torch

from src.models.attention import CausalSelfAttention


def test_causal_mask_prevents_future_leakage() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(
        d_model=16,
        n_heads=4,
        dropout=0.0,
        max_seq_len=16,
    )
    attention.eval()

    x = torch.randn(1, 8, 16)
    y = x.clone()
    y[:, 4:, :] = torch.randn_like(y[:, 4:, :])

    out_x, _ = attention(x, return_attention=True)
    out_y, _ = attention(y, return_attention=True)
    assert torch.allclose(out_x[:, :4, :], out_y[:, :4, :], atol=1e-5)
