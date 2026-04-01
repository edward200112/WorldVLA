# UniTok-Drive-Lite

一个最小可运行的自动驾驶 unified-token 原型。当前主干已经从 `facebook/chameleon-7b` 迁移到 `BAAI/Emu3-Chat-hf`，目标是在尽量少改动的前提下，保留 unified-token、LoRA-only、selective action mask、两阶段规划这些核心思路。

## 当前状态

- 主干模型：`BAAI/Emu3-Chat-hf`
- 微调方式：LoRA only，不做全量参数更新
- 低显存路径：支持 4-bit 量化加载
- 当前权威主链路：`scripts/` + `unitok_drive_lite/`
- 顶层 `data/ infer/ models/ train/`：保留为实验组件和独立 demo，不作为当前唯一真源

## 真实实现边界

当前代码里的 unified-token，不是“把所有模态都手工塞成同一串原生视觉 token”，而是下面这个更现实、也更可运行的版本：

- 文本、前视图图像、当前 BEV 图像：
  由 Emu3 的 `processor + chat template` 风格输入处理
- action / future BEV：
  先离散化为固定全局 special token，再接入共享语言 token 空间
- 历史动作摘要：
  作为单独的 summary token 区间接入
- 训练监督：
  只监督 future action token 和 future BEV token

也就是说，当前共享空间的核心是：

1. Emu3 负责文本和当前图像上下文建模
2. 动作与未来 BEV 通过固定离散 token 接入同一 LM 词表
3. 通过 selective attention mask 控制 future action token 的可见性

## Selective Action Mask

项目保留了原始 selective action mask 的设计思想：

- future action token 之间不能互相看见
- 但它们可以看到：
  - 文本 token
  - 当前前视图图像 token
  - 当前 BEV token
  - 历史动作 summary token

当前实现分两种接法：

- 训练 / 全序列前向：
  优先使用 4D additive mask，精确保留上述 selective 约束
- `generate(...)` 路径：
  如果底层 `transformers` / Emu3 版本对 4D mask 不稳定，会自动回退到 2D padding mask

因此，严格的 selective 推理更推荐使用项目里当前 planner 的“query 打分 / rollout”方式，而不是直接依赖裸 `generate(...)`。

## Token 设计

当前项目已经移除 Chameleon 时代那种 batch 内临时造词表的方式，改成全局固定 token registry：

- action token：固定全局区间
- future BEV token：固定全局区间
- history action summary token：固定全局区间
- 结构化标签 token：固定全局区间

这意味着同一个训练集、同一个推理流程、同一个 tokenizer 扩词顺序下，action / BEV token 的 id 保持一致。

## 项目边界

这是一个最小原型，不是完整自动驾驶栈，也不是可直接用于真实部署的系统。当前边界如下：

- `ToyUnifiedDriveDataset` 是合成 toy 数据，只用于跑通主流程
- planner 的 scorer 默认是启发式规则，不是 learned scorer
- future BEV 离散化目前是最小实现，不是高保真世界模型
- selective mask 在训练路径更准确，在 `generate(...)` 路径可能退化为近似实现
- 顶层 `train/train_sft.py`、`infer/planner.py`、`models/`、`data/` 主要用于独立实验和组件验证

## 唯一权威代码路径

当前建议把下面这条路径视为唯一主链路：

- `scripts/train_minimal.py`
- `scripts/run_demo.py`
- `unitok_drive_lite/`

如果后续继续迭代 Emu3 主线，优先修改这套代码，再考虑是否同步顶层实验目录。

## 目录概览

```text
WorldVLA/UniTok-Drive-Lite/
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_minimal.py
│   └── run_demo.py
├── unitok_drive_lite/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── discretizer.py
│   ├── masking.py
│   ├── model.py
│   ├── token_registry.py
│   └── train_utils.py
├── train/
│   └── train_sft.py
├── infer/
│   └── planner.py
├── models/
│   ├── action_tokenizer.py
│   ├── attention_mask.py
│   ├── backbone_chameleon.py
│   └── backbone_emu3.py
└── data/
    └── bev_rasterizer.py
```

## 环境准备

### Python

- 推荐 Python 3.10+

### 安装依赖

```bash
pip install -r requirements.txt
```

说明：

- 不再需要 Chameleon 的 gated 模型访问说明
- 不再需要围绕 `facebook/chameleon-7b` 做 Hugging Face 登录配置
- 如果你本地已经有 Hugging Face 通用缓存或镜像，直接按 Emu3 依赖安装即可

### 低显存说明

- Linux + CUDA 环境下可以使用 `bitsandbytes` 做 4-bit 量化
- CPU 或非 CUDA 环境也可以运行 mock demo / toy 逻辑，但不能指望实际 Emu3 大模型训练速度

## 运行方式

### 1. 最小训练闭环

这条路径使用包内 toy dataset，主要目的是跑通 `dataset -> discretizer -> model -> train_utils`：

```bash
python3 scripts/train_minimal.py --dataset_size 8 --num_epochs 1
```

训练完成后会保存到：

```bash
outputs/unitok_drive_lite/checkpoint_last
```

### 2. 最小推理演示

```bash
python3 scripts/run_demo.py --checkpoint_dir outputs/unitok_drive_lite/checkpoint_last
```

### 3. 两阶段 planner mock demo

这个 demo 不依赖真实 Emu3 权重，可以直接跑通“动作候选 -> future BEV rollout -> 打分 -> 轨迹解码”：

```bash
python3 infer/planner.py
```

### 4. 自定义 SFT 训练脚本

如果你已经有自己的样本文件，可以使用手写训练循环版本：

```bash
python3 train/train_sft.py --data_path path/to/your_samples.pt
```

这个脚本会：

- 用 Emu3 processor 编码文本 + 当前前视图 + 当前 BEV
- 监督 future action token
- 监督 future BEV token
- 叠加 action tokenizer 的重建损失

## 关键实现说明

### Emu3 输入组织

- 当前图像上下文优先交给 Emu3 processor 处理
- 不再复用 Chameleon 时代“只手工拼接图像 token”的做法
- future action / future BEV 作为离散 token 追加到目标序列

### LoRA 与新增 token

- 默认冻结底模
- 只训练 LoRA 参数
- 以及新增 special token 对应的 embedding / lm_head 行

### 两阶段 planner

当前 planner 已拆成四个核心函数：

- `generate_action_candidates(...)`
- `rollout_future_bev(...)`
- `score_candidates(...)`
- `plan_once(...)`

默认 scorer 是启发式实现，但已经预留自定义 learned scorer 接口。

## 已弃用 / 兼容说明

- `models/backbone_chameleon.py` 仍保留，但仅作旧实验参考
- 当前项目主线不再围绕 Chameleon 展开
- 如果文档、脚本或旧笔记里还出现 `facebook/chameleon-7b`，应视为历史信息
