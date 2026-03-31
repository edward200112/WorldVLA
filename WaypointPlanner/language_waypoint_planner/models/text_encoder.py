"""Text encoder with offline hash mode and optional HuggingFace backend."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from language_waypoint_planner.data.tokenizer import HashTextTokenizer


class HashTextBackbone(nn.Module):
    """Small transformer encoder operating on hash-tokenized text."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_length: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode token ids into contextualized text tokens."""

        embeddings = self.embedding(token_ids) + self.position_embedding[:, : token_ids.shape[1]]
        encoded = self.encoder(embeddings, src_key_padding_mask=~attention_mask)
        encoded = self.norm(encoded)
        return encoded


class TextEncoder(nn.Module):
    """Text encoder abstraction with a HuggingFace path and offline fallback."""

    def __init__(
        self,
        backend: str,
        output_dim: int,
        max_length: int,
        vocab_size: int,
        freeze: bool = False,
        model_name: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.max_length = max_length
        self.output_dim = output_dim
        self.hash_tokenizer = HashTextTokenizer(vocab_size=vocab_size)

        if backend == "hash":
            self.tokenizer = self.hash_tokenizer
            self.model = HashTextBackbone(
                vocab_size=vocab_size,
                hidden_dim=output_dim,
                max_length=max_length,
                dropout=dropout,
            )
            self.projection = nn.Identity()
        elif backend == "huggingface":
            if model_name is None:
                raise ValueError("model_name must be provided for HuggingFace backend")
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "transformers is required for backend='huggingface'. Install the optional dependency first."
                ) from exc
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
            hidden_size = int(getattr(self.model.config, "hidden_size"))
            self.projection = nn.Linear(hidden_size, output_dim) if hidden_size != output_dim else nn.Identity()
        else:
            raise ValueError(f"Unsupported text backend: {backend}")

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode a batch of text strings into fusion-ready tokens."""

        if self.backend == "hash":
            token_ids, attention_mask = self.hash_tokenizer.encode_batch(texts, self.max_length)
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            encoder_mask = attention_mask.clone()
            empty_rows = ~encoder_mask.any(dim=1)
            if empty_rows.any():
                encoder_mask[empty_rows, 0] = True
            encoded = self.model(token_ids, encoder_mask)
            encoded = encoded * attention_mask.unsqueeze(-1)
            return self.projection(encoded), attention_mask, {"token_ids": token_ids}

        encoded_inputs = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"].bool()
        for row, text in enumerate(texts):
            if not text.strip():
                attention_mask[row] = False
                input_ids[row] = getattr(self.tokenizer, "pad_token_id", 0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model_mask = attention_mask.clone()
        empty_rows = ~model_mask.any(dim=1)
        if empty_rows.any():
            model_mask[empty_rows, 0] = True
        model_outputs = self.model(input_ids=input_ids, attention_mask=model_mask.long())
        hidden = model_outputs.last_hidden_state
        hidden = hidden * attention_mask.unsqueeze(-1)
        return self.projection(hidden), attention_mask, {"token_ids": input_ids}
