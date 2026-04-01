from __future__ import annotations

"""Emu3 主干加载与前向辅助工具。

设计说明：
- Emu3 的多模态输入需要同时满足两件事：
  1. 文本 prompt 中保留 `<image>` 占位；
  2. 实际图像必须交给 `Emu3Processor` 处理成 `pixel_values` / `image_sizes`。
- 因此不能简单复用 Chameleon 时代那种“只靠手工拼 token id”的做法。
  对 Emu3 而言，图像占位 token 与视觉张量的对齐由 processor 负责，少了任一侧都不完整。
"""

import warnings
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .attention_mask import infer_padding_mask_from_additive_attention_mask
from unitok_drive_lite.token_registry import DEFAULT_DRIVE_SPECIAL_TOKENS

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Emu3ForConditionalGeneration, Emu3Processor
except ImportError:
    BitsAndBytesConfig = None
    Emu3ForConditionalGeneration = None
    Emu3Processor = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None



def _require_dependencies() -> None:
    """检查 Emu3 依赖是否已安装。"""
    if (
        BitsAndBytesConfig is None
        or Emu3ForConditionalGeneration is None
        or Emu3Processor is None
        or LoraConfig is None
        or TaskType is None
        or get_peft_model is None
        or prepare_model_for_kbit_training is None
    ):
        raise ImportError(
            "请先安装 transformers、peft、accelerate、bitsandbytes，并确保当前 transformers 版本包含 Emu3。"
        )


def _resolve_torch_dtype(torch_dtype: str | torch.dtype) -> torch.dtype:
    """把字符串或 torch.dtype 统一解析成 torch.dtype。"""
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if torch_dtype not in mapping:
        raise ValueError(f"不支持的 torch_dtype: {torch_dtype}")
    return mapping[torch_dtype]


def _find_first_parameter_device(model: nn.Module) -> torch.device:
    """推断模型当前主要设备。"""
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _find_first_floating_dtype(model: nn.Module) -> torch.dtype:
    """推断模型当前主要浮点 dtype。"""
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _deduplicate_tokens(tokens: Sequence[str]) -> list[str]:
    """去重并保持 special token 原始顺序。"""
    return list(dict.fromkeys(tokens))


def _freeze_all_parameters(model: nn.Module) -> None:
    """冻结模型全部参数。"""
    for parameter in model.parameters():
        parameter.requires_grad = False


def _collect_embedding_like_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    """收集输入 embedding 和输出 lm_head 对应的权重参数。"""
    collected_parameters: list[torch.nn.Parameter] = []
    visited_parameter_ids: set[int] = set()

    for getter_name in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, getter_name, None)
        if getter is None:
            continue
        module = getter()
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter) and id(weight) not in visited_parameter_ids:
            collected_parameters.append(weight)
            visited_parameter_ids.add(id(weight))

    for name, parameter in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if "embed_tokens" in name or "lm_head" in name:
            if id(parameter) not in visited_parameter_ids:
                collected_parameters.append(parameter)
                visited_parameter_ids.add(id(parameter))
    return collected_parameters


def _enable_only_new_token_rows(model: nn.Module, trainable_token_start: int) -> None:
    """只让新增 token 对应的 embedding 行参与训练。"""
    old_handles = getattr(model, "_new_token_gradient_hook_handles", None)
    if old_handles is not None:
        for handle in old_handles:
            handle.remove()

    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    target_parameters = _collect_embedding_like_parameters(model)

    for parameter in target_parameters:
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


def add_special_tokens(
    model: nn.Module | None,
    processor: Any,
    special_tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    """向 Emu3 tokenizer 与模型词表中添加自动驾驶 special tokens。

    说明：
    - Emu3 的图像占位 token 不需要这里手动添加，processor 会使用模型内置视觉占位机制。
    - 这里仅扩充自动驾驶领域自定义 token，例如 `<ACT>`、`<BEV>` 等。
    """
    _require_dependencies()

    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    requested_tokens = _deduplicate_tokens(special_tokens or DEFAULT_DRIVE_SPECIAL_TOKENS)
    old_vocab_size = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": requested_tokens})
    new_vocab_size = len(tokenizer)

    if model is not None and new_vocab_size != old_vocab_size:
        model.resize_token_embeddings(new_vocab_size)

    token_to_id = {token: tokenizer.convert_tokens_to_ids(token) for token in requested_tokens}
    token_info = {
        "requested_tokens": requested_tokens,
        "num_added_tokens": num_added_tokens,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "token_to_id": token_to_id,
    }

    if model is not None:
        previous_start = getattr(model, "_trainable_token_start", old_vocab_size)
        if new_vocab_size > old_vocab_size:
            trainable_token_start = min(previous_start, old_vocab_size)
        else:
            trainable_token_start = previous_start

        model._special_token_info = token_info
        model._special_token_ids = token_to_id
        model._trainable_token_start = trainable_token_start

        if trainable_token_start < new_vocab_size:
            _enable_only_new_token_rows(model, trainable_token_start)

    return token_info


def build_model_and_processor(
    model_name: str = "BAAI/Emu3-Chat-hf",
    load_in_4bit: bool = True,
    torch_dtype: str | torch.dtype = torch.bfloat16,
    device_map: str | dict[str, Any] | None = "auto",
    attn_implementation: str = "eager",
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    add_drive_special_tokens: bool = True,
    special_tokens: Sequence[str] | None = None,
    gradient_checkpointing: bool = False,
    token: str | None = None,
) -> tuple[nn.Module, Any]:
    """构建 Emu3 主干模型和对应 processor。

    关键配置说明：
    - `model_name` 默认使用 `BAAI/Emu3-Chat-hf`，这是文本生成和图文理解更合适的 checkpoint。
    - `processor.apply_chat_template(...)` 应优先用于构建对话 prompt，因为 Emu3 训练时依赖固定聊天格式。
    - 图像输入不能只手工拼接 `<image>` token；还必须把真实图像交给 `processor(images=..., text=...)`
      或 `processor.image_processor(...)` 生成视觉张量，再与文本 prompt 一起送入模型。
    - `load_in_4bit=True` 可用于低显存 LoRA 微调。
    - `use_lora=True` 时默认冻结底模，仅训练 LoRA 和新增 token 对应的 embedding 行。
    """
    _require_dependencies()

    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    processor = Emu3Processor.from_pretrained(model_name, token=token)
    processor.tokenizer.padding_side = "left"

    pretrained_kwargs: dict[str, Any] = {
        "attn_implementation": attn_implementation,
        "device_map": device_map,
        "token": token,
    }

    if load_in_4bit:
        pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolved_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        pretrained_kwargs["torch_dtype"] = resolved_dtype

    model = Emu3ForConditionalGeneration.from_pretrained(model_name, **pretrained_kwargs)

    special_token_info: dict[str, Any] | None = None
    if add_drive_special_tokens:
        special_token_info = add_special_tokens(model=model, processor=processor, special_tokens=special_tokens)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if use_lora:
        modules_to_save = ["embed_tokens", "lm_head"]

        if load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
            )
        else:
            _freeze_all_parameters(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=list(lora_target_modules),
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
    else:
        _freeze_all_parameters(model)

    if special_token_info is not None:
        trainable_token_start = special_token_info["old_vocab_size"]
        model._special_token_info = special_token_info
        model._special_token_ids = special_token_info["token_to_id"]
        model._trainable_token_start = trainable_token_start
        _enable_only_new_token_rows(model, trainable_token_start)

    return model, processor


def forward_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    **forward_kwargs: Any,
) -> Any:
    """执行一次 Emu3 前向传播，但不包含训练逻辑。

    约定：
    - 如果 `batch` 里已经包含 `pixel_values` / `image_sizes`，函数会自动搬运到模型设备。
    - 如果上层已经自己构造好了多模态 batch，本函数不改其字段结构，只做 dtype / device 适配。
    - 训练或 planner 的“全序列 query 打分”场景可以传 4D additive mask。
    - `generation_attention_mask` 是给 `generate(...)` 准备的旁路字段，这里会主动忽略。
    """
    target_device = _find_first_parameter_device(model)
    target_dtype = _find_first_floating_dtype(model)

    prepared_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "generation_attention_mask":
            continue
        if torch.is_tensor(value):
            tensor = value
            if tensor.device != target_device:
                if tensor.is_floating_point() and key == "pixel_values":
                    tensor = tensor.to(device=target_device, dtype=target_dtype)
                else:
                    tensor = tensor.to(device=target_device)
            elif tensor.is_floating_point() and key == "pixel_values":
                tensor = tensor.to(dtype=target_dtype)
            prepared_batch[key] = tensor
        else:
            prepared_batch[key] = value

    prepared_batch.update(forward_kwargs)
    return model(**prepared_batch)


def generate(
    model: nn.Module,
    batch: Mapping[str, Any],
    **generate_kwargs: Any,
) -> torch.Tensor:
    """执行 Emu3 的 generate，并自动处理设备与 dtype。

    说明：
    - Transformers 的 `generate(...)` 路径通常更稳定地支持 2D padding mask，而不是自定义 4D additive mask。
    - 如果上层误传了 4D selective mask，这里会自动退化为 2D padding mask。
      这种退化只保留“非 PAD 有效位”，不会保留 future action 互不可见的细粒度约束。
    - 因此严格的 selective action 推理，优先使用 `forward_batch(...)` 手工逐步解码或 query 打分。
    """
    target_device = _find_first_parameter_device(model)
    target_dtype = _find_first_floating_dtype(model)

    prepared_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            tensor = value
            if tensor.device != target_device:
                if tensor.is_floating_point() and key == "pixel_values":
                    tensor = tensor.to(device=target_device, dtype=target_dtype)
                else:
                    tensor = tensor.to(device=target_device)
            elif tensor.is_floating_point() and key == "pixel_values":
                tensor = tensor.to(dtype=target_dtype)
            prepared_batch[key] = tensor
        else:
            prepared_batch[key] = value

    generation_attention_mask = prepared_batch.pop("generation_attention_mask", None)
    attention_mask = prepared_batch.get("attention_mask")
    if torch.is_tensor(attention_mask) and attention_mask.ndim == 4:
        warnings.warn(
            "Emu3 generate 路径收到 4D selective attention mask，已自动回退到 2D padding mask。"
            "如果需要严格的 future action selective visibility，请改用 forward_batch(...)。",
            stacklevel=2,
        )
        if generation_attention_mask is None:
            generation_attention_mask = infer_padding_mask_from_additive_attention_mask(attention_mask)
        prepared_batch["attention_mask"] = generation_attention_mask.to(device=target_device)
    elif generation_attention_mask is not None and "attention_mask" not in prepared_batch:
        prepared_batch["attention_mask"] = generation_attention_mask.to(device=target_device)

    return model.generate(**prepared_batch, **generate_kwargs)
