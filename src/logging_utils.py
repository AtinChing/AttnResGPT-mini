from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from src.utils import ensure_dir, timestamp


@dataclass
class RunPaths:
    run_dir: Path
    checkpoint_dir: Path
    probe_dir: Path
    train_log_path: Path
    val_log_path: Path
    summary_path: Path
    resolved_config_path: Path
    tokenizer_path: Path


def create_run_paths(output_root: str | Path, experiment_name: str, resume_from: Optional[str]) -> RunPaths:
    if resume_from is not None:
        run_dir = Path(resume_from)
    else:
        run_dir = Path(output_root) / f"{experiment_name}_{timestamp()}"
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    probe_dir = ensure_dir(run_dir / "probes")
    ensure_dir(run_dir)
    return RunPaths(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        probe_dir=probe_dir,
        train_log_path=run_dir / "train_metrics.jsonl",
        val_log_path=run_dir / "val_metrics.jsonl",
        summary_path=run_dir / "run_summary.json",
        resolved_config_path=run_dir / "resolved_config.yaml",
        tokenizer_path=run_dir / "tokenizer.json",
    )


class ExperimentLogger:
    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def log_train(self, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(self.paths.train_log_path, payload)

    def log_val(self, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(self.paths.val_log_path, payload)

    def save_probe(self, step: int, payload: Mapping[str, Any]) -> Path:
        probe_path = self.paths.probe_dir / f"step_{step:07d}.pt"
        torch.save(dict(payload), probe_path)
        return probe_path

    def save_checkpoint(self, step: int, payload: Mapping[str, Any]) -> Path:
        checkpoint_path = self.paths.checkpoint_dir / f"step_{step:07d}.pt"
        torch.save(dict(payload), checkpoint_path)
        return checkpoint_path

    def prune_old_checkpoints(self, keep_last_k: int) -> None:
        if keep_last_k <= 0:
            return
        checkpoints = sorted(self.paths.checkpoint_dir.glob("step_*.pt"))
        stale = checkpoints[:-keep_last_k]
        for checkpoint in stale:
            checkpoint.unlink(missing_ok=True)

    def save_summary(self, payload: Mapping[str, Any]) -> None:
        with self.paths.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


def checkpoint_payload(
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    best_val_loss: Optional[float],
) -> dict[str, Any]:
    return {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }
