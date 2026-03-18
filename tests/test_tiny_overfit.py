from __future__ import annotations

import pytest
import torch

from src.config import AttnResConfig, ModelConfig
from src.data.tokenizer_utils import CharTokenizer
from src.metrics import language_model_loss
from src.models.gpt_attnres import GPTAttnRes
from src.models.gpt_baseline import GPTBaseline


def make_config(architecture: str, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        vocab_size=vocab_size,
        max_seq_len=24,
        d_model=48,
        n_layers=2,
        n_heads=4,
        d_ff=96,
        dropout=0.0,
        attnres=AttnResConfig(enabled=architecture == "attnres", final_readout=True),
    )


def run_tiny_overfit(model: torch.nn.Module, input_ids: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    initial_loss = None
    final_loss = None
    for step in range(60):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids, return_aux=False)
        loss = language_model_loss(logits, targets)
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    return float(initial_loss), float(final_loss)


@pytest.mark.slow
def test_tiny_overfit_baseline_and_attnres() -> None:
    text = ("task=copy input=abcabc target=abcabc\n" * 32).strip()
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor(token_ids[:24], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(token_ids[1:25], dtype=torch.long).unsqueeze(0)

    baseline = GPTBaseline(make_config("baseline", tokenizer.vocab_size))
    attnres = GPTAttnRes(make_config("attnres", tokenizer.vocab_size))

    baseline_initial, baseline_final = run_tiny_overfit(baseline, input_ids, targets)
    attn_initial, attn_final = run_tiny_overfit(attnres, input_ids, targets)

    assert baseline_final < baseline_initial
    assert attn_final < attn_initial
    assert baseline_final < 1.5
    assert attn_final < 1.5
