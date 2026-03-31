"""Lightweight deterministic tokenizer used for offline text experiments."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch


_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


@dataclass
class HashTextTokenizer:
    """Simple hash-based tokenizer with fixed vocabulary size."""

    vocab_size: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size must be at least 8")

    def normalize(self, text: str) -> str:
        """Normalize user-provided text before tokenization."""

        return " ".join(_TOKEN_PATTERN.findall(text.lower().strip()))

    def _hash_token(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).hexdigest()
        numeric = int(digest, 16)
        return 3 + numeric % (self.vocab_size - 3)

    def encode_text(self, text: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single string into token ids and attention mask."""

        if max_length < 1:
            raise ValueError("max_length must be positive")
        normalized = self.normalize(text)
        if not normalized:
            ids = [self.pad_token_id] * max_length
            mask = [0] * max_length
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)
        hashed = [self._hash_token(token) for token in normalized.split()]
        token_ids = hashed[: max_length - 1] + [self.eos_token_id]
        pad_len = max_length - len(token_ids)
        ids = token_ids + [self.pad_token_id] * pad_len
        mask = [1] * len(token_ids) + [0] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def encode_batch(self, texts: Sequence[str], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of strings into tensors."""

        encoded = [self.encode_text(text, max_length) for text in texts]
        ids = torch.stack([item[0] for item in encoded], dim=0)
        masks = torch.stack([item[1] for item in encoded], dim=0)
        return ids, masks

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode token ids into a coarse textual representation."""

        words: List[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in (self.pad_token_id, self.bos_token_id):
                continue
            if token_id == self.eos_token_id:
                break
            words.append(f"tok{token_id}")
        return " ".join(words).strip()
