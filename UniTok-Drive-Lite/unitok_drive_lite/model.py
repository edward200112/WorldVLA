from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import warnings

import torch
from torch import nn

from .config import ExperimentConfig
from .discretizer import UnifiedDriveDiscretizer
from .masking import build_selective_attention_mask
from .token_registry import TokenRegistry

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Emu3ForConditionalGeneration
except ImportError:
    AutoProcessor = None
    BitsAndBytesConfig = None
    Emu3ForConditionalGeneration = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


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


def _freeze_all_parameters(model: nn.Module) -> None:
    """冻结模型全部参数。"""
    for parameter in model.parameters():
        parameter.requires_grad = False


def _collect_embedding_like_parameters(model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """收集输入 embedding 和输出 lm_head 对应的参数。"""
    collected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if "embed_tokens" in name or "lm_head" in name:
            collected.append((name, parameter))
    return collected


def _enable_only_new_token_rows(model: nn.Module, trainable_token_start: int) -> None:
    """只让新增 token 行参与训练，旧词表行梯度会被清零。"""
    old_handles = getattr(model, "_new_token_gradient_hook_handles", None)
    if old_handles is not None:
        for handle in old_handles:
            handle.remove()

    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    for _, parameter in _collect_embedding_like_parameters(model):
        parameter.requires_grad = True

        def _mask_old_rows(grad: torch.Tensor, start_index: int = trainable_token_start) -> torch.Tensor:
            """把旧词表行的梯度归零，仅保留新增 token 行。"""
            masked_grad = grad.clone()
            masked_grad[:start_index, ...] = 0
            return masked_grad

        hook_handles.append(parameter.register_hook(_mask_old_rows))

    model._new_token_gradient_hook_handles = hook_handles
    model._trainable_token_start = trainable_token_start


def _is_attention_mask_compatibility_error(error: Exception) -> bool:
    """判断异常是否来自 4D attention mask 兼容性。"""
    message = str(error).lower()
    return "attention_mask" in message or "4d" in message or "mask" in message


class UnifiedDriveModel(nn.Module):
    """最小版 unified-token 自动驾驶模型包装器。"""

    def __init__(self, config: ExperimentConfig) -> None:
        """加载 Emu3 主干、LoRA、固定全局 token 和统一离散器。"""
        super().__init__()
        if AutoProcessor is None or Emu3ForConditionalGeneration is None or get_peft_model is None:
            raise ImportError("请先安装 transformers、peft、accelerate，再初始化 UnifiedDriveModel。")

        self.config = config
        self.compute_dtype = resolve_torch_dtype(config.model.torch_dtype)
        self.processor = AutoProcessor.from_pretrained(config.model.model_name)
        # Emu3 官方建议 batched generation 时使用 left padding，
        # 这是最小兼容设置，不影响训练逻辑（训练时 batch_size=1 逐条前向）。
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.processor.tokenizer
        self.token_registry = TokenRegistry.from_token_config(config.tokens)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        old_vocab_size = len(self.tokenizer)
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.token_registry.all_special_tokens})
        self.trainable_token_start = old_vocab_size

        pretrained_kwargs: Dict[str, Any] = {
            "torch_dtype": self.compute_dtype,
            "attn_implementation": config.model.attn_implementation,
        }
        if config.model.load_in_4bit:
            if BitsAndBytesConfig is None or prepare_model_for_kbit_training is None:
                raise ImportError("4-bit 加载需要额外安装 bitsandbytes 和 peft。")
            pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        backbone = Emu3ForConditionalGeneration.from_pretrained(
            config.model.model_name,
            **pretrained_kwargs,
        )
        if len(self.tokenizer) != old_vocab_size:
            backbone.resize_token_embeddings(len(self.tokenizer))
        backbone.config.use_cache = False

        if config.model.load_in_4bit:
            backbone = prepare_model_for_kbit_training(backbone)

        _freeze_all_parameters(backbone)

        lora_config = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=list(config.model.lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.backbone = get_peft_model(backbone, lora_config)
        if len(self.tokenizer) > self.trainable_token_start:
            _enable_only_new_token_rows(self.backbone, self.trainable_token_start)

        self.discretizer = UnifiedDriveDiscretizer(
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=config.tokens,
        )
        self._supports_custom_4d_mask = True

    @property
    def device(self) -> torch.device:
        """返回当前模型所在设备。"""
        return next(self.parameters()).device

    def _forward_single_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """对单条样本执行前向，必要时回退到标准 2D attention mask。"""
        sample_attention_mask: torch.Tensor
        if self.config.model.use_selective_attention_mask and self._supports_custom_4d_mask:
            sample_attention_mask = build_selective_attention_mask(
                attention_mask=attention_mask[0],
                token_types=token_types,
                dtype=next(self.backbone.parameters()).dtype,
            )
        else:
            sample_attention_mask = attention_mask

        try:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=sample_attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                use_cache=False,
                return_dict=True,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            if not self._supports_custom_4d_mask or not _is_attention_mask_compatibility_error(error):
                raise
            self._supports_custom_4d_mask = False
            warnings.warn(
                "当前 Emu3 / Transformers 版本不接受 4D selective attention mask，"
                "已自动回退到标准 2D causal/padding mask。"
                "这会丢失 future action 互不可见的精确约束，只适合作为兼容性回退。",
                stacklevel=2,
            )
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                use_cache=False,
                return_dict=True,
            )
        return outputs.logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values_list: list[torch.Tensor],
        image_sizes_list: list[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """执行一次前向传播并注入 selective attention mask。"""
        logits_list: list[torch.Tensor] = []
        for batch_index in range(input_ids.shape[0]):
            sample_logits = self._forward_single_sample(
                input_ids=input_ids[batch_index : batch_index + 1],
                attention_mask=attention_mask[batch_index : batch_index + 1],
                token_types=token_types[batch_index],
                pixel_values=pixel_values_list[batch_index].to(device=self.device, dtype=self.compute_dtype),
                image_sizes=image_sizes_list[batch_index].to(device=self.device),
            )
            logits_list.append(sample_logits)

        logits = torch.cat(logits_list, dim=0)
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_function(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"logits": logits, "loss": loss}

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

    def adapter_state_dict(self) -> Dict[str, Any]:
        """只导出 LoRA 参数和新增 token 行。"""
        state_dict = self.backbone.state_dict()
        new_token_rows: Dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if "embed_tokens.weight" in name or "lm_head.weight" in name:
                new_token_rows[name] = tensor[self.trainable_token_start :].detach().cpu()

        return {
            "lora": {
                name: tensor.detach().cpu()
                for name, tensor in state_dict.items()
                if "lora_" in name
            },
            "new_token_rows": new_token_rows,
            "trainable_token_start": self.trainable_token_start,
        }

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        """保存最小检查点。"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter_state_dict(), checkpoint_dir / "adapter_state.pt")
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(checkpoint_dir / "processor")

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        """加载最小检查点。"""
        checkpoint_path = checkpoint_dir / "adapter_state.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        lora_state = checkpoint.get("lora", checkpoint)
        self.backbone.load_state_dict(lora_state, strict=False)

        start_index = int(checkpoint.get("trainable_token_start", self.trainable_token_start))
        named_parameters = dict(self.backbone.named_parameters())
        for name, rows in checkpoint.get("new_token_rows", {}).items():
            parameter = named_parameters.get(name)
            if parameter is None:
                continue
            parameter.data[start_index : start_index + rows.shape[0]].copy_(
                rows.to(device=parameter.device, dtype=parameter.dtype)
            )
