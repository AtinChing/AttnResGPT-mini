from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class AttnResConfig:
    enabled: bool = False
    window_size: Optional[int] = None
    temperature: float = 1.0
    rmsnorm_keys: bool = True
    zero_init_queries: bool = True
    include_embedding: bool = True
    keep_embedding_in_window: bool = True
    final_readout: bool = True


@dataclass
class ModelConfig:
    architecture: str = "baseline"
    vocab_size: int = 0
    max_seq_len: int = 128
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True
    norm_eps: float = 1e-5
    init_std: float = 0.02
    attnres: AttnResConfig = field(default_factory=AttnResConfig)


@dataclass
class SyntheticDataConfig:
    num_train_sequences: int = 4096
    num_val_sequences: int = 512
    min_pattern_length: int = 5
    max_pattern_length: int = 18
    task_mix: list[str] = field(
        default_factory=lambda: ["copy", "reverse", "sort", "shift", "interleave"]
    )


@dataclass
class DataConfig:
    dataset_type: str = "synthetic"
    tokenizer_type: str = "char"
    text_path: Optional[str] = None
    train_text_path: Optional[str] = None
    val_text_path: Optional[str] = None
    block_size: int = 128
    batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 2
    pin_memory: bool = True
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)


@dataclass
class TrainingConfig:
    max_steps: int = 1000
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"
    compile_model: bool = False
    log_interval: int = 10
    eval_interval: int = 100
    probe_interval: int = 100
    checkpoint_interval: int = 250
    eval_max_batches: Optional[int] = None
    device: str = "auto"
    resume_from: Optional[str] = None


@dataclass
class LoggingConfig:
    output_root: str = "runs"
    save_probes: bool = True
    save_checkpoints: bool = True
    keep_last_k_checkpoints: int = 2


@dataclass
class EvalConfig:
    max_batches: Optional[int] = None
    run_ablation: bool = True
    ablation_fractions: list[float] = field(default_factory=lambda: [0.25, 0.5])


@dataclass
class ExperimentConfig:
    name: str = "attnres_experiment"
    seed: int = 1337
    deterministic: bool = False
    notes: str = ""


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


def _construct_config(values: dict[str, Any]) -> Config:
    return Config(
        experiment=ExperimentConfig(**values.get("experiment", {})),
        data=DataConfig(
            **{
                **values.get("data", {}),
                "synthetic": SyntheticDataConfig(**values.get("data", {}).get("synthetic", {})),
            }
        ),
        model=ModelConfig(
            **{
                **values.get("model", {}),
                "attnres": AttnResConfig(**values.get("model", {}).get("attnres", {})),
            }
        ),
        training=TrainingConfig(**values.get("training", {})),
        logging=LoggingConfig(**values.get("logging", {})),
        evaluation=EvalConfig(**values.get("evaluation", {})),
    )


def config_to_dict(config: Config) -> dict[str, Any]:
    return asdict(config)


def apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        key, raw_value = override.split("=", maxsplit=1)
        value = yaml.safe_load(raw_value)
        cursor = config_dict
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return config_dict


def validate_config(config: Config) -> Config:
    if config.model.architecture not in {"baseline", "attnres"}:
        raise ValueError(f"Unsupported architecture: {config.model.architecture}")
    if config.model.d_model % config.model.n_heads != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads")
    if config.data.block_size > config.model.max_seq_len:
        raise ValueError("data.block_size must be <= model.max_seq_len")
    if config.model.architecture == "attnres":
        config.model.attnres.enabled = True
    if config.training.min_lr > config.training.learning_rate:
        raise ValueError("training.min_lr must be <= training.learning_rate")
    if config.data.dataset_type not in {"synthetic", "text", "tinystories"}:
        raise ValueError("data.dataset_type must be one of synthetic, text, tinystories")
    return config


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if overrides:
        payload = apply_overrides(payload, overrides)
    config = _construct_config(payload)
    return validate_config(config)


def save_config(config: Config, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
