from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    unk_token: str = "<unk>"

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = [cls.unk_token] + sorted(set(text))
        stoi = {token: index for index, token in enumerate(vocab)}
        return cls(stoi=stoi, itos=vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(char, unk_id) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[index] for index in ids)

    def save(self, path: str | Path) -> None:
        payload = {"stoi": self.stoi, "itos": self.itos, "unk_token": self.unk_token}
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            stoi={str(key): int(value) for key, value in payload["stoi"].items()},
            itos=[str(value) for value in payload["itos"]],
            unk_token=str(payload.get("unk_token", "<unk>")),
        )


def build_tokenizer(tokenizer_type: str, text: str) -> CharTokenizer:
    if tokenizer_type != "char":
        raise ValueError(
            "This research codebase intentionally keeps tokenization simple and local-first. "
            "Supported tokenizer_type: char."
        )
    return CharTokenizer.from_text(text)
