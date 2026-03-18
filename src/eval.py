from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch

from src.config import Config, load_config
from src.data.dataset import build_dataloaders
from src.metrics import average_dicts, language_model_loss, perplexity_from_loss
from src.models.gpt_attnres import GPTAttnRes
from src.models.gpt_baseline import GPTBaseline
from src.utils import amp_dtype_from_string, count_parameters, get_device


def build_model(config: Config) -> torch.nn.Module:
    if config.model.architecture == "baseline":
        return GPTBaseline(config.model)
    if config.model.architecture == "attnres":
        return GPTAttnRes(config.model)
    raise ValueError(f"Unsupported architecture: {config.model.architecture}")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: Optional[int] = None,
    ablate_sublayers: Optional[set[int]] = None,
    return_aux: bool = False,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    aux_summaries: list[dict[str, float]] = []
    use_autocast = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        autocast_context = torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_autocast,
        )
        with autocast_context if use_autocast else nullcontext():
            logits, aux = model(
                input_ids,
                ablate_sublayers=ablate_sublayers,
                return_aux=return_aux,
            )
            loss = language_model_loss(logits, targets)
        losses.append(float(loss.item()))
        if return_aux:
            summary = {
                key: float(value)
                for key, value in aux.items()
                if isinstance(value, (int, float))
            }
            if summary:
                aux_summaries.append(summary)

    mean_loss = sum(losses) / max(1, len(losses))
    metrics = {
        "val_loss": mean_loss,
        "val_perplexity": perplexity_from_loss(mean_loss),
    }
    metrics.update(average_dicts(aux_summaries))
    return metrics


def build_ablation_set(num_sublayers: int, fraction: float, which: str) -> set[int]:
    count = max(1, int(round(num_sublayers * fraction)))
    if which == "early":
        return set(range(count))
    if which == "late":
        return set(range(num_sublayers - count, num_sublayers))
    raise ValueError("which must be 'early' or 'late'")


@torch.no_grad()
def run_ablation_sweep(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    fractions: list[float],
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    num_sublayers = getattr(model, "num_sublayers")
    results: dict[str, float] = {}
    for fraction in fractions:
        for which in ("early", "late"):
            ablate_set = build_ablation_set(num_sublayers, fraction, which)
            metrics = evaluate_model(
                model,
                dataloader,
                device=device,
                amp_dtype=amp_dtype,
                max_batches=max_batches,
                ablate_sublayers=ablate_set,
                return_aux=False,
            )
            prefix = f"ablation_{which}_{int(fraction * 100):02d}"
            results[f"{prefix}_loss"] = metrics["val_loss"]
            results[f"{prefix}_perplexity"] = metrics["val_perplexity"]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pt file.")
    parser.add_argument("--overrides", nargs="*", default=[], help="Optional key=value config overrides.")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--run-ablation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides)
    tokenizer, _, val_loader, data_meta = build_dataloaders(config)
    config.model.vocab_size = tokenizer.vocab_size

    device = get_device(config.training.device)
    amp_dtype = amp_dtype_from_string(config.training.amp_dtype)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    metrics = evaluate_model(
        model,
        val_loader,
        device=device,
        amp_dtype=amp_dtype,
        max_batches=args.max_batches or config.evaluation.max_batches,
        return_aux=True,
    )
    if args.run_ablation:
        metrics.update(
            run_ablation_sweep(
                model,
                val_loader,
                device=device,
                amp_dtype=amp_dtype,
                fractions=config.evaluation.ablation_fractions,
                max_batches=args.max_batches or config.evaluation.max_batches,
            )
        )

    counts = count_parameters(model)
    metrics.update({f"params_{key}": float(value) for key, value in counts.items()})
    metrics.update({f"data_{key}": float(value) for key, value in data_meta.items()})
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
