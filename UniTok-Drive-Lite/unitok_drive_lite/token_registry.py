from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TokenConfig


BASE_DRIVE_SPECIAL_TOKENS: tuple[str, ...] = (
    "<BEV>",
    "<ACT>",
    "<PLAN>",
    "<DREAM>",
    "<ACT_SUMMARY>",
)

STRUCTURED_TEXT_LABEL_TOKENS: tuple[str, ...] = (
    "<NAV_LEFT>",
    "<NAV_RIGHT>",
    "<NAV_STRAIGHT>",
    "<LIGHT_RED>",
    "<LIGHT_GREEN>",
    "<RISK_PED>",
    "<RISK_VEH>",
    "<RISK_OCC>",
)

DEFAULT_DRIVE_SPECIAL_TOKENS: tuple[str, ...] = BASE_DRIVE_SPECIAL_TOKENS + STRUCTURED_TEXT_LABEL_TOKENS


def build_fixed_token_strings(prefix: str, count: int) -> list[str]:
    """根据固定前缀生成一组全局离散 token 字符串。"""
    if count < 0:
        raise ValueError("固定 token 数量不能小于 0。")
    if count == 0:
        return []
    width = max(4, len(str(count - 1)))
    return [f"<{prefix}_{index:0{width}d}>" for index in range(count)]


def _resolve_token_id(tokenizer: Any, token: str) -> int:
    """把 special token 字符串解析成稳定 token id。"""
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
        raise ValueError(f"tokenizer 无法识别固定 token: {token}")
    return int(token_id)


@dataclass(frozen=True)
class ResolvedTokenRegistry:
    """保存 token registry 在具体 tokenizer 下解析出的固定 id。"""

    special_token_ids: dict[str, int]
    structured_label_token_ids: dict[str, int]
    action_token_ids: list[int]
    bev_token_ids: list[int]
    summary_token_ids: list[int]


@dataclass(frozen=True)
class TokenRegistry:
    """统一维护项目里所有固定 token 区间。

    说明：
    - 这里不在 batch 内临时构造 token 词表。
    - action / BEV / summary token 的字符串和注册顺序在整个训练、推理期间保持不变。
    - 真正的 token id 由同一底模 tokenizer 按固定顺序扩词后确定，因此整个项目内部是一致的。
    """

    special_tokens: tuple[str, ...]
    structured_label_tokens: tuple[str, ...]
    action_tokens: tuple[str, ...]
    bev_tokens: tuple[str, ...]
    summary_tokens: tuple[str, ...]

    @classmethod
    def from_token_config(cls, config: TokenConfig) -> "TokenRegistry":
        """从包内 TokenConfig 构造 registry。"""
        return cls(
            special_tokens=BASE_DRIVE_SPECIAL_TOKENS,
            structured_label_tokens=STRUCTURED_TEXT_LABEL_TOKENS,
            action_tokens=tuple(build_fixed_token_strings(config.action_token_prefix, config.action_codebook_size)),
            bev_tokens=tuple(build_fixed_token_strings(config.bev_token_prefix, config.bev_codebook_size)),
            summary_tokens=tuple(build_fixed_token_strings(config.summary_token_prefix, config.summary_codebook_size)),
        )

    @classmethod
    def from_vocab_sizes(
        cls,
        action_vocab_size: int,
        bev_vocab_size: int,
        summary_vocab_size: int = 0,
        action_token_prefix: str = "UT_ACT",
        bev_token_prefix: str = "UT_BEV",
        summary_token_prefix: str = "UT_SUM",
    ) -> "TokenRegistry":
        """从显式词表大小构造 registry，供顶层训练/推理脚本复用。"""
        return cls(
            special_tokens=BASE_DRIVE_SPECIAL_TOKENS,
            structured_label_tokens=STRUCTURED_TEXT_LABEL_TOKENS,
            action_tokens=tuple(build_fixed_token_strings(action_token_prefix, action_vocab_size)),
            bev_tokens=tuple(build_fixed_token_strings(bev_token_prefix, bev_vocab_size)),
            summary_tokens=tuple(build_fixed_token_strings(summary_token_prefix, summary_vocab_size)),
        )

    @property
    def all_special_tokens(self) -> list[str]:
        """返回需要一次性注册进 tokenizer 的全部固定 token。"""
        return list(
            self.special_tokens
            + self.structured_label_tokens
            + self.action_tokens
            + self.bev_tokens
            + self.summary_tokens
        )

    def action_index_to_token(self, index: int) -> str:
        """把 action 离散索引映射到固定 token 字符串。"""
        if index < 0 or index >= len(self.action_tokens):
            raise ValueError(f"action 索引越界: index={index}, vocab_size={len(self.action_tokens)}")
        return self.action_tokens[index]

    def bev_index_to_token(self, index: int) -> str:
        """把 BEV 离散索引映射到固定 token 字符串。"""
        if index < 0 or index >= len(self.bev_tokens):
            raise ValueError(f"BEV 索引越界: index={index}, vocab_size={len(self.bev_tokens)}")
        return self.bev_tokens[index]

    def summary_index_to_token(self, index: int) -> str:
        """把 summary 离散索引映射到固定 token 字符串。"""
        if index < 0 or index >= len(self.summary_tokens):
            raise ValueError(f"summary 索引越界: index={index}, vocab_size={len(self.summary_tokens)}")
        return self.summary_tokens[index]

    def action_indices_to_tokens(self, indices: list[int]) -> list[str]:
        """把 action 索引序列映射成固定 token 字符串序列。"""
        return [self.action_index_to_token(index) for index in indices]

    def bev_indices_to_tokens(self, indices: list[int]) -> list[str]:
        """把 BEV 索引序列映射成固定 token 字符串序列。"""
        return [self.bev_index_to_token(index) for index in indices]

    def resolve_tokenizer(self, tokenizer: Any) -> ResolvedTokenRegistry:
        """在具体 tokenizer 下解析固定 token id 区间。"""
        return ResolvedTokenRegistry(
            special_token_ids={token: _resolve_token_id(tokenizer, token) for token in self.special_tokens},
            structured_label_token_ids={
                token: _resolve_token_id(tokenizer, token) for token in self.structured_label_tokens
            },
            action_token_ids=[_resolve_token_id(tokenizer, token) for token in self.action_tokens],
            bev_token_ids=[_resolve_token_id(tokenizer, token) for token in self.bev_tokens],
            summary_token_ids=[_resolve_token_id(tokenizer, token) for token in self.summary_tokens],
        )
