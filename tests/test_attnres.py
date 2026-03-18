from __future__ import annotations

import torch

from src.config import AttnResConfig, ModelConfig
from src.models.attnres import DepthAttentionResidual
from src.models.gpt_attnres import GPTAttnRes


def test_attnres_weights_sum_to_one() -> None:
    module = DepthAttentionResidual(
        d_model=8,
        temperature=1.0,
        window_size=None,
        zero_init_query=True,
        include_embedding=True,
        keep_embedding_in_window=True,
    )
    history = [torch.randn(2, 4, 8) for _ in range(5)]
    _, stats = module(history, return_stats=True)
    total = stats["mean_weights"].sum().item()
    assert abs(total - 1.0) < 1e-5
    assert stats["source_indices"] == [0, 1, 2, 3, 4]


def test_sliding_window_keeps_embedding() -> None:
    module = DepthAttentionResidual(
        d_model=8,
        window_size=2,
        zero_init_query=True,
        include_embedding=True,
        keep_embedding_in_window=True,
    )
    history = [torch.randn(1, 3, 8) for _ in range(6)]
    _, stats = module(history, return_stats=True)
    assert stats["source_indices"] == [0, 4, 5]


def test_attnres_activation_norms_are_finite_at_init() -> None:
    config = ModelConfig(
        architecture="attnres",
        vocab_size=24,
        max_seq_len=12,
        d_model=24,
        n_layers=2,
        n_heads=4,
        d_ff=48,
        dropout=0.0,
        attnres=AttnResConfig(enabled=True, final_readout=True),
    )
    model = GPTAttnRes(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    _, aux = model(input_ids, return_aux=True)
    assert all(torch.isfinite(torch.tensor(value)) for value in aux["block_output_norms"])
    assert torch.isfinite(torch.tensor(aux["depth_attention_entropy"]))
