from __future__ import annotations

import torch

from src.config import AttnResConfig, ModelConfig
from src.models.gpt_attnres import GPTAttnRes
from src.models.gpt_baseline import GPTBaseline


def make_config(architecture: str) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        vocab_size=32,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        attnres=AttnResConfig(enabled=architecture == "attnres", final_readout=True),
    )


def test_baseline_logits_shape() -> None:
    model = GPTBaseline(make_config("baseline"))
    input_ids = torch.randint(0, 32, (2, 16))
    logits, aux = model(input_ids, return_aux=True)
    assert logits.shape == (2, 16, 32)
    assert len(aux["block_output_norms"]) == 2


def test_attnres_logits_shape() -> None:
    model = GPTAttnRes(make_config("attnres"))
    input_ids = torch.randint(0, 32, (2, 16))
    logits, aux = model(input_ids, return_aux=True)
    assert logits.shape == (2, 16, 32)
    assert len(aux["block_output_norms"]) == 2
    assert len(aux["depth_attention"]) == 2 * 2 + 1
