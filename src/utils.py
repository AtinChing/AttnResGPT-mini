from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def format_count(value: int) -> str:
    thresholds = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]
    for threshold, suffix in thresholds:
        if value >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return str(value)


def overall_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)


def cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def cosine_warmup_multiplier(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def amp_dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported AMP dtype: {name}")
    return mapping[key]
