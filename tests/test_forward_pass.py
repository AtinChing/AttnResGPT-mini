from __future__ import annotations

import torch

from src.config import AttnResConfig, ModelConfig
from src.utils import count_parameters
from src.models.gpt_attnres import GPTAttnRes
from src.models.gpt_baseline import GPTBaseline


def make_config(architecture: str) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        vocab_size=40,
        max_seq_len=20,
        d_model=48,
        n_layers=3,
        n_heads=4,
        d_ff=96,
        dropout=0.0,
        attnres=AttnResConfig(enabled=architecture == "attnres", final_readout=True),
    )


def test_parameter_counts_are_similar() -> None:
    baseline = GPTBaseline(make_config("baseline"))
    attnres = GPTAttnRes(make_config("attnres"))
    baseline_params = count_parameters(baseline)["total"]
    attnres_params = count_parameters(attnres)["total"]
    delta_pct = abs(attnres_params - baseline_params) / baseline_params
    assert attnres_params > baseline_params
    assert delta_pct < 0.05


def test_forward_aux_contains_depth_metrics_for_attnres() -> None:
    model = GPTAttnRes(make_config("attnres"))
    input_ids = torch.randint(0, model.config.vocab_size, (2, 20))
    _, aux = model(input_ids, return_aux=True)
    assert "early_contribution" in aux
    assert "late_contribution" in aux
    assert "embedding_contribution" in aux
