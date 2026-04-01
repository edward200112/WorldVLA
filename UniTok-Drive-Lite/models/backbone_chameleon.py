from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        BitsAndBytesConfig,
        ChameleonForConditionalGeneration,
        ChameleonProcessor,
    )
except ImportError:
    BitsAndBytesConfig = None
    ChameleonForConditionalGeneration = None
    ChameleonProcessor = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


DEFAULT_DRIVE_SPECIAL_TOKENS: tuple[str, ...] = (
    "<BEV>",
    "<ACT>",
    "<PLAN>",
    "<DREAM>",
    "<ACT_SUMMARY>",
    "<NAV_LEFT>",
    "<NAV_RIGHT>",
    "<NAV_STRAIGHT>",
    "<LIGHT_RED>",
    "<LIGHT_GREEN>",
    "<RISK_PED>",
    "<RISK_VEH>",
    "<RISK_OCC>",
)


def _require_dependencies() -> None:
    """检查 Transformers 和 PEFT 依赖是否已安装。"""
    if (
        BitsAndBytesConfig is None
        or ChameleonForConditionalGeneration is None
        or ChameleonProcessor is None
        or LoraConfig is None
        or TaskType is None
        or get_peft_model is None
        or prepare_model_for_kbit_training is None
    ):
        raise ImportError(
            "请先安装 transformers、peft、accelerate、bitsandbytes，再使用 Chameleon backbone。"
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
    """去重并保持 special token 的原始顺序。"""
    return list(dict.fromkeys(tokens))


def _freeze_all_parameters(model: nn.Module) -> None:
    """冻结模型全部参数。"""
    for parameter in model.parameters():
        parameter.requires_grad = False


def _collect_embedding_like_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    """收集输入 embedding 和输出 lm_head 对应的权重参数。

    这里优先通过官方接口获取，如果遇到 PEFT 包装，再回退到名称匹配。
    """
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

    name_patterns = ("embed_tokens", "lm_head")
    for name, parameter in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if any(pattern in name for pattern in name_patterns):
            if id(parameter) not in visited_parameter_ids:
                collected_parameters.append(parameter)
                visited_parameter_ids.add(id(parameter))
    return collected_parameters


def _enable_only_new_token_rows(model: nn.Module, trainable_token_start: int) -> None:
    """只让新增 token 对应的 embedding 行参与训练。

    说明：
    - PyTorch 不能直接对同一个 Parameter 的部分行设置 `requires_grad`
    - 因此这里通过 gradient hook 把旧词表行的梯度清零
    - 这样最终只有新增 token 的 embedding / lm_head 行会被更新
    """
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
    """向 processor tokenizer 与模型词表中添加自动驾驶 special tokens。

    参数：
    - model: 已加载的 Chameleon 模型，也可以为 None；如果为 None，则只改 tokenizer
    - processor: ChameleonProcessor
    - special_tokens: 要添加的 special token 列表；默认使用内置自动驾驶 token

    返回：
    - 一个字典，包含新增 token 数量、词表大小变化与 token-id 映射
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
        trainable_token_start = min(previous_start, old_vocab_size) if new_vocab_size > old_vocab_size else previous_start

        model._special_token_info = token_info
        model._special_token_ids = token_to_id
        model._trainable_token_start = trainable_token_start

        # 如果此时模型已经完成 LoRA 包装，这一步会重新挂上“仅训练新增 token 行”的梯度掩码。
        if trainable_token_start < new_vocab_size:
            _enable_only_new_token_rows(model, trainable_token_start)

    return token_info


def build_model_and_processor(
    model_name: str = "facebook/chameleon-7b",
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
    """构建 Chameleon 主干模型和对应 processor。

    关键配置说明：
    - `load_in_4bit=True`:
      使用 bitsandbytes 的 4-bit 量化加载底模，适合 QLoRA 式微调，显著节省显存。
    - `attn_implementation="eager"`:
      默认使用 eager attention，便于后续传入自定义 4D selective attention mask。
    - `use_lora=True`:
      默认只训练 LoRA 适配器；底模参数保持冻结。
    - `add_drive_special_tokens=True`:
      默认会把自动驾驶统一 token 中常用的 special tokens 预留进词表。
    - “只训练新增 token embedding” 的实现方式：
      这里通过梯度掩码只更新新词表行，而不是更新整个 embedding 矩阵。
    """
    _require_dependencies()

    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    processor = ChameleonProcessor.from_pretrained(model_name, token=token)

    quantization_config = None
    pretrained_kwargs: dict[str, Any] = {
        "attn_implementation": attn_implementation,
        "device_map": device_map,
        "token": token,
    }

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolved_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        pretrained_kwargs["quantization_config"] = quantization_config
    else:
        pretrained_kwargs["torch_dtype"] = resolved_dtype

    model = ChameleonForConditionalGeneration.from_pretrained(model_name, **pretrained_kwargs)

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

    # LoRA 包装后重新恢复“新增 token 行可训练”的能力。
    trainable_token_start = None
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
    """执行一次前向传播，但不包含训练逻辑。

    用法约定：
    - `batch` 通常来自 `processor(..., return_tensors="pt")`
    - 也可以来自你自己的 collator，只要字段名符合 Hugging Face 模型 forward 接口
    - 如果 batch 中包含 `pixel_values`，会自动转换到模型的主要浮点 dtype
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

    prepared_batch.update(forward_kwargs)
    return model(**prepared_batch)
