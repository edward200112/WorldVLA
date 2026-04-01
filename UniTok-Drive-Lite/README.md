# UniTok-Drive-Lite

一个最小可运行的自动驾驶 unified-token 原型，目标是把导航文本、前视图图像、BEV 栅格、动作都映射到同一个离散 token 空间，再用 `facebook/chameleon-7b` 做自回归建模。

## 设计边界

- 主干模型固定为 Hugging Face `facebook/chameleon-7b`
- 仅使用 LoRA 微调，不做全量参数更新
- 图像、BEV、动作都先离散成 token，再进入同一个 token 序列
- 未来预测目标包含：
  - future action token
  - future 3 帧 BEV token
- 通过自定义 4D attention mask 实现 selective attention：
  - 当前 action chunk 内的 raw action token 彼此不可见
  - 但它们可以看到文本、前视图图像、当前 BEV、历史动作 summary token

## 目录

```text
WorldVLA/UniTok-Drive-Lite/
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_minimal.py
│   └── run_demo.py
└── unitok_drive_lite/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── discretizer.py
    ├── masking.py
    ├── model.py
    └── train_utils.py
```

## 环境准备

1. 使用 Python 3.10+
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. `facebook/chameleon-7b` 是 gated 模型，需要先在 Hugging Face 页面接受协议并登录：

```bash
huggingface-cli login
```

## 运行方式

### 训练最小样例

```bash
python3 scripts/train_minimal.py
```

### 运行推理演示

```bash
python3 scripts/run_demo.py --checkpoint_dir outputs/unitok_drive_lite/checkpoint_last
```

## 实现说明

- 为了严格满足 “LoRA only”，代码没有扩展 Chameleon 词表，而是复用 Chameleon 现有的 `<reservedXXXXX>` token 作为统一离散空间。
- 前视图图像和 BEV 采用轻量、确定性的 patch 平均量化，动作采用 2D 连续控制离散化。
- 历史动作会被压缩成 1 个 summary token。
- 为了严格避免 action token 彼此泄漏，训练阶段对 future action 与 future BEV 使用 query 占位 token，并在同一位置上计算监督损失，而不是使用标准 shift loss。
- 自定义 attention mask 依赖 `attn_implementation="eager"`，因为这样可以直接传入 4D additive mask。

## 资源提示

- `facebook/chameleon-7b` 需要较大的显存或内存。
- 默认配置只是原型闭环，训练数据使用合成 toy 数据集，优先保证主流程可读、可改、可跑通。
