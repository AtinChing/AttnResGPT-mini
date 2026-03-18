from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.data.tokenizer_utils import CharTokenizer, build_tokenizer


SYMBOLS = "abcdefgijklmnopqrstwxyz0123456789"


class TokenSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
        super().__init__()
        if len(token_ids) <= block_size:
            raise ValueError("Corpus is too small for the configured block size")
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = torch.tensor(self.token_ids[index : index + self.block_size], dtype=torch.long)
        y = torch.tensor(self.token_ids[index + 1 : index + 1 + self.block_size], dtype=torch.long)
        return {"input_ids": x, "targets": y}


def _random_symbols(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(SYMBOLS) for _ in range(length))


def _shift_symbol(char: str) -> str:
    index = SYMBOLS.index(char)
    return SYMBOLS[(index + 1) % len(SYMBOLS)]


def _interleave(left: str, right: str) -> str:
    result = []
    for a, b in zip(left, right):
        result.extend([a, b])
    if len(left) > len(right):
        result.extend(list(left[len(right) :]))
    elif len(right) > len(left):
        result.extend(list(right[len(left) :]))
    return "".join(result)


def _build_synthetic_example(rng: random.Random, task: str, length: int) -> str:
    source = _random_symbols(rng, length)
    if task == "copy":
        target = source
    elif task == "reverse":
        target = source[::-1]
    elif task == "sort":
        target = "".join(sorted(source))
    elif task == "shift":
        target = "".join(_shift_symbol(char) for char in source)
    elif task == "interleave":
        midpoint = max(1, length // 2)
        target = _interleave(source[:midpoint], source[midpoint:])
    else:
        raise ValueError(f"Unsupported synthetic task: {task}")
    return f"task={task} input={source} target={target}\n"


def generate_synthetic_corpus(config: Config, *, split: str) -> str:
    synthetic = config.data.synthetic
    num_sequences = (
        synthetic.num_train_sequences if split == "train" else synthetic.num_val_sequences
    )
    rng = random.Random(config.experiment.seed + (0 if split == "train" else 10_000))
    lines = []
    for _ in range(num_sequences):
        task = rng.choice(synthetic.task_mix)
        length = rng.randint(synthetic.min_pattern_length, synthetic.max_pattern_length)
        lines.append(_build_synthetic_example(rng, task, length))
    return "".join(lines)


def _extract_text_fields(payload: Any) -> Iterable[str]:
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            yield from _extract_text_fields(item)
    elif isinstance(payload, dict):
        for key in ("text", "story", "content", "prompt", "completion"):
            if key in payload:
                yield from _extract_text_fields(payload[key])


def _read_json_file(path: Path) -> str:
    if path.suffix == ".jsonl":
        texts = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                texts.extend(_extract_text_fields(json.loads(line)))
        return "\n".join(texts)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return "\n".join(_extract_text_fields(payload))


def read_local_corpus(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find corpus at {resolved}")
    if resolved.is_file():
        if resolved.suffix in {".json", ".jsonl"}:
            return _read_json_file(resolved)
        return resolved.read_text(encoding="utf-8")

    texts: list[str] = []
    for file_path in sorted(resolved.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix in {".txt", ".md"}:
            texts.append(file_path.read_text(encoding="utf-8"))
        elif file_path.suffix in {".json", ".jsonl"}:
            texts.append(_read_json_file(file_path))
    if not texts:
        raise ValueError(f"No readable text files found under {resolved}")
    return "\n".join(texts)


def _split_train_val_from_text(text: str) -> tuple[str, str]:
    cutoff = max(1, int(0.9 * len(text)))
    train_text = text[:cutoff]
    val_text = text[cutoff:]
    if len(val_text) < 128:
        tail = min(len(text), 512)
        train_text = text[:-tail]
        val_text = text[-tail:]
    return train_text, val_text


def build_datasets(
    config: Config,
    tokenizer: Optional[CharTokenizer] = None,
) -> tuple[CharTokenizer, TokenSequenceDataset, TokenSequenceDataset, dict[str, Any]]:
    if config.data.dataset_type == "synthetic":
        train_text = generate_synthetic_corpus(config, split="train")
        val_text = generate_synthetic_corpus(config, split="val")
    else:
        if config.data.train_text_path:
            train_text = read_local_corpus(config.data.train_text_path)
            val_path = config.data.val_text_path or config.data.train_text_path
            val_text = read_local_corpus(val_path)
        elif config.data.text_path:
            train_text, val_text = _split_train_val_from_text(read_local_corpus(config.data.text_path))
        else:
            raise ValueError("A text_path or train_text_path must be provided for text-based datasets")

    tokenizer = tokenizer or build_tokenizer(config.data.tokenizer_type, train_text + val_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    train_dataset = TokenSequenceDataset(train_ids, config.data.block_size)
    val_dataset = TokenSequenceDataset(val_ids, config.data.block_size)
    metadata = {
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
    }
    return tokenizer, train_dataset, val_dataset, metadata


def build_dataloaders(
    config: Config,
    tokenizer: Optional[CharTokenizer] = None,
) -> tuple[CharTokenizer, DataLoader, DataLoader, dict[str, Any]]:
    tokenizer, train_dataset, val_dataset, metadata = build_datasets(config, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )
    return tokenizer, train_loader, val_loader, metadata
