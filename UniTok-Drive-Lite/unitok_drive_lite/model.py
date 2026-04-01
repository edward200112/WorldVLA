from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from .config import ExperimentConfig
from .discretizer import UnifiedDriveDiscretizer
from .masking import build_selective_attention_mask

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, ChameleonForConditionalGeneration
except ImportError:
    AutoTokenizer = None
    ChameleonForConditionalGeneration = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """把字符串形式的 dtype 解析成 torch.dtype。"""
    mapping: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 torch_dtype: {dtype_name}")
    return mapping[dtype_name]


class UnifiedDriveModel(nn.Module):
    """最小版 unified-token 自动驾驶模型包装器。"""

    def __init__(self, config: ExperimentConfig) -> None:
        """加载 Chameleon 主干、LoRA 和统一离散器。"""
        super().__init__()
        if AutoTokenizer is None or ChameleonForConditionalGeneration is None or get_peft_model is None:
            raise ImportError("请先安装 transformers 与 peft，再初始化 UnifiedDriveModel。")

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.discretizer = UnifiedDriveDiscretizer(self.tokenizer, config.tokens)

        torch_dtype = resolve_torch_dtype(config.model.torch_dtype)
        backbone = ChameleonForConditionalGeneration.from_pretrained(
            config.model.model_name,
            torch_dtype=torch_dtype,
            attn_implementation=config.model.attn_implementation,
        )
        backbone.config.use_cache = False

        lora_config = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=list(config.model.lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.backbone = get_peft_model(backbone, lora_config)

    @property
    def device(self) -> torch.device:
        """返回当前模型所在设备。"""
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """执行一次前向传播并注入 selective attention mask。"""
        additive_mask = build_selective_attention_mask(
            attention_mask=attention_mask,
            role_ids=role_ids,
            dtype=next(self.backbone.parameters()).dtype,
        )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=additive_mask,
            use_cache=False,
            return_dict=True,
        )
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_function(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        return {"logits": outputs.logits, "loss": loss}

    def count_trainable_parameters(self) -> tuple[int, int]:
        """统计总参数量与可训练参数量。"""
        total_parameters = 0
        trainable_parameters = 0
        for parameter in self.parameters():
            count = parameter.numel()
            total_parameters += count
            if parameter.requires_grad:
                trainable_parameters += count
        return total_parameters, trainable_parameters

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """只导出 LoRA 适配器参数。"""
        state_dict = self.backbone.state_dict()
        return {
            name: tensor.detach().cpu()
            for name, tensor in state_dict.items()
            if "lora_" in name
        }

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        """保存最小检查点。"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter_state_dict(), checkpoint_dir / "adapter_state.pt")

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        """加载最小检查点。"""
        checkpoint_path = checkpoint_dir / "adapter_state.pt"
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=False)
