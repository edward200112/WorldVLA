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
            if masked_grad.ndim == 1:
                masked_grad[:start_index] = 0
            else:
                masked_grad[:start_index, ...] = 0
            return masked_grad

        hook_handles.append(parameter.register_hook(_mask_old_rows))

    model._new_token_gradient_hook_handles = hook_handles
    model._trainable_token_start = trainable_token_start


def _is_attention_mask_compatibility_error(error: Exception) -> bool:
    """判断异常是否来自 4D attention mask 兼容性。"""
    message = str(error).lower()
    return "attention_mask" in message or "4d" in message or "mask" in message


def _extract_linear_vocab_size(module: Any) -> int | None:
    """从可能的 lm_head / output head 模块中提取输出词表大小。"""
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
        return int(weight.shape[0])
    out_features = getattr(module, "out_features", None)
    if out_features is not None:
        return int(out_features)
    return None


def _resize_linear_output_head(module: nn.Module, new_vocab_size: int) -> nn.Module:
    """扩展线性输出头到新的词表大小，并保留旧权重。"""
    if not isinstance(module, nn.Linear):
        raise TypeError(f"暂不支持为 {type(module)} 自动扩展 lm_head。")

    old_vocab_size = int(module.weight.shape[0])
    if old_vocab_size == new_vocab_size:
        return module

    new_head = nn.Linear(
        module.in_features,
        new_vocab_size,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        new_head.weight[:old_vocab_size].copy_(module.weight)
        if new_vocab_size > old_vocab_size:
            nn.init.normal_(
                new_head.weight[old_vocab_size:],
                mean=float(module.weight.mean().item()),
                std=float(module.weight.std().item()) if module.weight.numel() > 1 else 0.02,
            )
        if module.bias is not None and new_head.bias is not None:
            new_head.bias[:old_vocab_size].copy_(module.bias)
            if new_vocab_size > old_vocab_size:
                new_head.bias[old_vocab_size:].zero_()
    return new_head


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
        self.token_registry.assert_unique_tokens()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenizer_vocab_size_before = len(self.tokenizer)
        num_added_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.token_registry.all_special_tokens}
        )
        tokenizer_vocab_size_after = len(self.tokenizer)
        self.trainable_token_start = tokenizer_vocab_size_before
        self._token_debug_header_printed = False
        self._tokenizer_resize_debug = {
            "tokenizer_vocab_size_before": tokenizer_vocab_size_before,
            "num_added_tokens": int(num_added_tokens),
            "tokenizer_vocab_size_after": tokenizer_vocab_size_after,
        }

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
        self._sync_model_vocab_with_tokenizer(backbone)
        backbone.config.use_cache = False

        if config.model.load_in_4bit:
            backbone = prepare_model_for_kbit_training(backbone)

        _freeze_all_parameters(backbone)

        lora_config = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=list(config.model.lora_target_modules),
            modules_to_save=["embed_tokens", "lm_head"],
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
        self.token_registry.assert_tokenizer_alignment(
            tokenizer=self.tokenizer,
            vocab_size=self.get_logits_vocab_size(),
        )
        self._supports_custom_4d_mask = True

    def get_input_embedding_vocab_size(self) -> int:
        """返回输入 embedding 的词表大小。"""
        input_embeddings = self.backbone.get_input_embeddings()
        if input_embeddings is None or getattr(input_embeddings, "weight", None) is None:
            raise RuntimeError("模型缺少输入 embedding 权重。")
        return int(input_embeddings.weight.shape[0])

    def get_lm_head_vocab_size(self) -> int | None:
        """返回 lm_head 词表大小；如果没有暴露输出 embedding，则返回 None。"""
        direct_lm_head = getattr(self.backbone, "lm_head", None)
        direct_lm_head_vocab = _extract_linear_vocab_size(direct_lm_head)
        if direct_lm_head_vocab is not None:
            return direct_lm_head_vocab
        output_embeddings = self.backbone.get_output_embeddings()
        if output_embeddings is None or getattr(output_embeddings, "weight", None) is None:
            return None
        return int(output_embeddings.weight.shape[0])

    def get_logits_vocab_size(self) -> int:
        """返回用于监督的 logits 词表大小。"""
        lm_head_vocab_size = self.get_lm_head_vocab_size()
        if lm_head_vocab_size is not None:
            return lm_head_vocab_size
        return self.get_input_embedding_vocab_size()

    def _sync_model_vocab_with_tokenizer(self, backbone: nn.Module) -> None:
        """确保 tokenizer、输入 embedding 和输出 head 的词表大小一致。"""
        tokenizer_vocab_size = len(self.tokenizer)
        input_embeddings = backbone.get_input_embeddings()
        input_vocab_size = (
            None if input_embeddings is None or getattr(input_embeddings, "weight", None) is None
            else int(input_embeddings.weight.shape[0])
        )
        output_embeddings = backbone.get_output_embeddings()
        output_vocab_size = (
            None if output_embeddings is None or getattr(output_embeddings, "weight", None) is None
            else int(output_embeddings.weight.shape[0])
        )
        direct_lm_head = getattr(backbone, "lm_head", None)
        direct_lm_head_vocab_size = _extract_linear_vocab_size(direct_lm_head)
        self._tokenizer_resize_debug.update(
            {
                "input_vocab_size_before_resize": input_vocab_size,
                "output_vocab_size_before_resize": output_vocab_size,
                "direct_lm_head_vocab_size_before_resize": direct_lm_head_vocab_size,
            }
        )

        if input_vocab_size != tokenizer_vocab_size or (
            output_vocab_size is not None and output_vocab_size != tokenizer_vocab_size
        ):
            backbone.resize_token_embeddings(tokenizer_vocab_size)
            if hasattr(backbone, "tie_weights"):
                backbone.tie_weights()

        direct_lm_head = getattr(backbone, "lm_head", None)
        direct_lm_head_vocab_size = _extract_linear_vocab_size(direct_lm_head)
        if direct_lm_head_vocab_size is not None and direct_lm_head_vocab_size != tokenizer_vocab_size:
            resized_lm_head = _resize_linear_output_head(direct_lm_head, tokenizer_vocab_size)
            if hasattr(backbone, "set_output_embeddings"):
                try:
                    backbone.set_output_embeddings(resized_lm_head)
                except Exception:
                    backbone.lm_head = resized_lm_head
            else:
                backbone.lm_head = resized_lm_head

        # Emu3ForConditionalGeneration 的 forward/loss 使用 config.text_config.vocab_size
        # 和/或 config.vocab_size；如果这里只扩 embedding 而不更新这些元数据，
        # logits 与 loss 仍会沿用旧词表大小。
        if hasattr(backbone, "config") and backbone.config is not None:
            if hasattr(backbone.config, "vocab_size"):
                backbone.config.vocab_size = tokenizer_vocab_size
            text_config = getattr(backbone.config, "text_config", None)
            if text_config is not None and hasattr(text_config, "vocab_size"):
                text_config.vocab_size = tokenizer_vocab_size
        nested_model = getattr(backbone, "model", None)
        if nested_model is not None and hasattr(nested_model, "config") and nested_model.config is not None:
            if hasattr(nested_model.config, "vocab_size"):
                nested_model.config.vocab_size = tokenizer_vocab_size
            text_config = getattr(nested_model.config, "text_config", None)
            if text_config is not None and hasattr(text_config, "vocab_size"):
                text_config.vocab_size = tokenizer_vocab_size
            text_model = getattr(nested_model, "text_model", None)
            if text_model is not None:
                if hasattr(text_model, "vocab_size"):
                    text_model.vocab_size = tokenizer_vocab_size
                if hasattr(text_model, "config") and text_model.config is not None and hasattr(
                    text_model.config, "vocab_size"
                ):
                    text_model.config.vocab_size = tokenizer_vocab_size

        input_embeddings = backbone.get_input_embeddings()
        output_embeddings = backbone.get_output_embeddings()
        input_vocab_size_after = (
            None if input_embeddings is None or getattr(input_embeddings, "weight", None) is None
            else int(input_embeddings.weight.shape[0])
        )
        output_vocab_size_after = (
            None if output_embeddings is None or getattr(output_embeddings, "weight", None) is None
            else int(output_embeddings.weight.shape[0])
        )
        direct_lm_head_after = getattr(backbone, "lm_head", None)
        direct_lm_head_vocab_size_after = _extract_linear_vocab_size(direct_lm_head_after)
        self._tokenizer_resize_debug.update(
            {
                "input_vocab_size_after_resize": input_vocab_size_after,
                "output_vocab_size_after_resize": output_vocab_size_after,
                "direct_lm_head_vocab_size_after_resize": direct_lm_head_vocab_size_after,
                "config_vocab_size_after_resize": getattr(getattr(backbone, "config", None), "vocab_size", None),
                "config_text_vocab_size_after_resize": getattr(
                    getattr(getattr(backbone, "config", None), "text_config", None),
                    "vocab_size",
                    None,
                ),
            }
        )

        if input_vocab_size_after != tokenizer_vocab_size:
            raise RuntimeError(
                "resize_token_embeddings 后输入 embedding 词表仍与 tokenizer 不一致: "
                f"tokenizer={tokenizer_vocab_size} input_vocab={input_vocab_size_after}"
            )
        if output_vocab_size_after is not None and output_vocab_size_after != tokenizer_vocab_size:
            raise RuntimeError(
                "resize_token_embeddings 后 lm_head 词表仍与 tokenizer 不一致: "
                f"tokenizer={tokenizer_vocab_size} lm_head_vocab={output_vocab_size_after}"
            )
        if direct_lm_head_vocab_size_after is not None and direct_lm_head_vocab_size_after != tokenizer_vocab_size:
            raise RuntimeError(
                "显式 lm_head 扩词后仍与 tokenizer 不一致: "
                f"tokenizer={tokenizer_vocab_size} lm_head_vocab={direct_lm_head_vocab_size_after}"
            )

    def _format_token_from_id(self, token_id: int) -> str:
        """把 token id 转成人可读字符串。"""
        if token_id < 0:
            return f"<negative:{token_id}>"
        if token_id >= len(self.tokenizer):
            return f"<out-of-tokenizer-range:{token_id}>"
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        return str(token)

    def _print_label_debug_header(self, logits_vocab_size: int) -> None:
        """首次训练前打印 tokenizer / model 词表对齐信息。"""
        if self._token_debug_header_printed:
            return
        print(
            "[debug:vocab] "
            f"len(tokenizer)={len(self.tokenizer)} "
            f"input_embedding_vocab={self.get_input_embedding_vocab_size()} "
            f"lm_head_vocab={self.get_lm_head_vocab_size()} "
            f"logits_vocab={logits_vocab_size} "
            f"trainable_token_start={self.trainable_token_start} "
            f"num_added_tokens={self._tokenizer_resize_debug['num_added_tokens']}"
        )
        print(f"[debug:vocab] resize_info={self._tokenizer_resize_debug}")
        self._token_debug_header_printed = True

    def _validate_labels_before_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """在 loss 前做显式标签范围检查，并打印必要调试信息。"""
        logits_vocab_size = int(logits.size(-1))
        tokenizer_vocab_size = len(self.tokenizer)
        input_vocab_size = self.get_input_embedding_vocab_size()
        lm_head_vocab_size = self.get_lm_head_vocab_size()
        self._print_label_debug_header(logits_vocab_size)
        self.token_registry.assert_tokenizer_alignment(
            tokenizer=self.tokenizer,
            vocab_size=logits_vocab_size,
        )
        if tokenizer_vocab_size != input_vocab_size:
            raise RuntimeError(
                f"tokenizer 与输入 embedding 词表不一致: tokenizer={tokenizer_vocab_size}, "
                f"input_embedding_vocab={input_vocab_size}"
            )
        if lm_head_vocab_size is not None and lm_head_vocab_size != logits_vocab_size:
            raise RuntimeError(
                f"lm_head 词表与 logits 不一致: lm_head_vocab={lm_head_vocab_size}, logits_vocab={logits_vocab_size}"
            )

        detached_labels = labels.detach()
        valid_mask = detached_labels.ne(-100)
        invalid_low_mask = detached_labels.lt(-100)
        invalid_high_mask = valid_mask & detached_labels.ge(logits_vocab_size)
        valid_labels = detached_labels[valid_mask]

        min_label = int(valid_labels.min().item()) if valid_labels.numel() > 0 else None
        max_label = int(valid_labels.max().item()) if valid_labels.numel() > 0 else None
        print(
            "[debug:labels] "
            f"valid_label_count={int(valid_mask.sum().item())} "
            f"min_valid_label={min_label} "
            f"max_valid_label={max_label} "
            f"has_label_lt_-100={bool(invalid_low_mask.any().item())} "
            f"has_label_ge_logits_vocab={bool(invalid_high_mask.any().item())}"
        )

        if invalid_low_mask.any() or invalid_high_mask.any():
            offending_mask = invalid_low_mask | invalid_high_mask
            offending_positions = torch.nonzero(offending_mask, as_tuple=False)[:16]
            offending_records: list[dict[str, Any]] = []
            for position in offending_positions.tolist():
                batch_index, token_index = position
                label_value = int(detached_labels[batch_index, token_index].item())
                input_id_value = int(input_ids[batch_index, token_index].item())
                offending_records.append(
                    {
                        "batch_index": batch_index,
                        "token_index": token_index,
                        "label_id": label_value,
                        "label_token": self._format_token_from_id(label_value),
                        "input_id": input_id_value,
                        "input_token": self._format_token_from_id(input_id_value),
                    }
                )
            flat_offending_ids = detached_labels[offending_mask][:16].tolist()
            offending_tokens = [self._format_token_from_id(int(token_id)) for token_id in flat_offending_ids]
            raise RuntimeError(
                "训练标签超出有效范围。"
                f" tokenizer={tokenizer_vocab_size}, input_vocab={input_vocab_size}, "
                f"lm_head_vocab={lm_head_vocab_size}, logits_vocab={logits_vocab_size}, "
                f"min_valid_label={min_label}, max_valid_label={max_label}, "
                f"label_lt_-100={bool(invalid_low_mask.any().item())}, "
                f"label_ge_logits_vocab={bool(invalid_high_mask.any().item())}, "
                f"offending_ids={flat_offending_ids}, offending_tokens={offending_tokens}, "
                f"offending_positions={offending_records}"
            )

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
            self._validate_labels_before_loss(input_ids, labels, logits)
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
