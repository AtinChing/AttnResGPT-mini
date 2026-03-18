from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from typing import Any

import torch
from torch.optim import AdamW

from src.config import load_config, save_config
from src.data.dataset import build_dataloaders
from src.eval import build_model, evaluate_model, run_ablation_sweep
from src.hooks import NormHookCollector
from src.logging_utils import ExperimentLogger, checkpoint_payload, create_run_paths
from src.metrics import language_model_loss, perplexity_from_loss
from src.utils import (
    amp_dtype_from_string,
    count_parameters,
    cycle,
    format_count,
    get_device,
    overall_grad_norm,
    seed_everything,
)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    min_lr_ratio = min_lr / max(base_lr, 1e-12)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def latest_probe_payload(
    *,
    step: int,
    hooks: NormHookCollector,
    aux: dict[str, Any],
) -> dict[str, Any]:
    depth_attention = []
    for row_index, weights in enumerate(aux.get("depth_attention", [])):
        source_indices = aux.get("source_indices", [])[row_index]
        depth_attention.append(
            {
                "row": row_index,
                "source_indices": source_indices,
                "mean_weights": weights.tolist(),
            }
        )
    return {
        "step": step,
        "activation_norms": dict(hooks.activation_norms),
        "gradient_norms": dict(hooks.gradient_norms),
        "depth_attention": depth_attention,
        "scalar_aux": {
            key: float(value)
            for key, value in aux.items()
            if isinstance(value, (int, float))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline GPT or AttnRes GPT.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--overrides", nargs="*", default=[], help="Optional key=value config overrides.")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides)
    seed_everything(config.experiment.seed, deterministic=config.experiment.deterministic)

    run_paths = create_run_paths(
        output_root=config.logging.output_root,
        experiment_name=config.experiment.name,
        resume_from=config.training.resume_from,
    )
    logger = ExperimentLogger(run_paths)

    tokenizer, train_loader, val_loader, data_meta = build_dataloaders(config)
    tokenizer.save(run_paths.tokenizer_path)
    config.model.vocab_size = tokenizer.vocab_size
    save_config(config, run_paths.resolved_config_path)

    device = get_device(config.training.device)
    amp_dtype = amp_dtype_from_string(config.training.amp_dtype)
    model = build_model(config).to(device)

    if config.training.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    counts = count_parameters(model)
    print(f"Run directory: {run_paths.run_dir}")
    print(
        "Parameters: "
        f"total={format_count(counts['total'])} "
        f"trainable={format_count(counts['trainable'])}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.max_steps,
        base_lr=config.training.learning_rate,
        min_lr=config.training.min_lr,
    )
    use_grad_scaler = (
        device.type == "cuda"
        and config.training.mixed_precision
        and amp_dtype == torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    hooks = NormHookCollector()
    hooks.register(model)

    start_step = 0
    best_val_loss: float | None = None
    if config.training.resume_from is not None:
        checkpoints = sorted(run_paths.checkpoint_dir.glob("step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under {run_paths.checkpoint_dir}")
        checkpoint = torch.load(checkpoints[-1], map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        best_val_loss = checkpoint.get("best_val_loss")

    train_iterator = cycle(train_loader)
    use_autocast = device.type == "cuda" and config.training.mixed_precision

    try:
        for step in range(start_step + 1, config.training.max_steps + 1):
            model.train()
            hooks.reset_step()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            last_aux: dict[str, Any] = {}
            probe_step = config.logging.save_probes and (
                step == 1
                or step == config.training.max_steps
                or step % config.training.probe_interval == 0
            )

            for _ in range(config.training.grad_accum_steps):
                batch = next(train_iterator)
                input_ids = batch["input_ids"].to(device)
                targets = batch["targets"].to(device)
                autocast_context = torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_autocast,
                )
                with autocast_context if use_autocast else nullcontext():
                    logits, aux = model(input_ids, return_aux=probe_step)
                    loss = language_model_loss(logits, targets)
                    scaled_loss = loss / config.training.grad_accum_steps
                total_loss += float(loss.item())
                last_aux = aux
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            grad_norm = overall_grad_norm(model)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            train_payload = {
                "step": step,
                "train_loss": total_loss / config.training.grad_accum_steps,
                "train_perplexity": perplexity_from_loss(total_loss / config.training.grad_accum_steps),
                "learning_rate": current_lr,
                "global_grad_norm": grad_norm,
            }
            train_payload.update(
                {
                    key: float(value)
                    for key, value in last_aux.items()
                    if isinstance(value, (int, float))
                }
            )
            if step % config.training.log_interval == 0 or step == 1:
                logger.log_train(train_payload)
                print(
                    f"step={step:05d} "
                    f"loss={train_payload['train_loss']:.4f} "
                    f"ppl={train_payload['train_perplexity']:.2f} "
                    f"lr={current_lr:.6f}"
                )

            if probe_step:
                logger.save_probe(step, latest_probe_payload(step=step, hooks=hooks, aux=last_aux))

            if step % config.training.eval_interval == 0 or step == config.training.max_steps:
                val_metrics = evaluate_model(
                    model,
                    val_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    max_batches=config.training.eval_max_batches,
                    return_aux=True,
                )
                val_payload = {"step": step, **val_metrics}
                logger.log_val(val_payload)
                print(
                    f"[val] step={step:05d} "
                    f"loss={val_payload['val_loss']:.4f} "
                    f"ppl={val_payload['val_perplexity']:.2f}"
                )
                if best_val_loss is None or val_payload["val_loss"] < best_val_loss:
                    best_val_loss = val_payload["val_loss"]

            if config.logging.save_checkpoints and (
                step % config.training.checkpoint_interval == 0 or step == config.training.max_steps
            ):
                logger.save_checkpoint(
                    step,
                    checkpoint_payload(
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if scaler.is_enabled() else None,
                        best_val_loss=best_val_loss,
                    ),
                )
                logger.prune_old_checkpoints(config.logging.keep_last_k_checkpoints)
    finally:
        hooks.close()

    final_eval = evaluate_model(
        model,
        val_loader,
        device=device,
        amp_dtype=amp_dtype,
        max_batches=config.evaluation.max_batches or config.training.eval_max_batches,
        return_aux=True,
    )
    ablation = {}
    if config.evaluation.run_ablation:
        ablation = run_ablation_sweep(
            model,
            val_loader,
            device=device,
            amp_dtype=amp_dtype,
            fractions=config.evaluation.ablation_fractions,
            max_batches=config.evaluation.max_batches or config.training.eval_max_batches,
        )

    summary = {
        "experiment_name": config.experiment.name,
        "architecture": config.model.architecture,
        "seed": config.experiment.seed,
        "parameter_count_total": counts["total"],
        "parameter_count_trainable": counts["trainable"],
        "best_val_loss": best_val_loss,
        **data_meta,
        **final_eval,
        **ablation,
    }
    logger.save_summary(summary)


if __name__ == "__main__":
    main()
