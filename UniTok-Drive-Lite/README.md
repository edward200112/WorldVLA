# UniTok-Drive-Lite

一个最小可运行的自动驾驶 unified-token 原型。主干模型为 `BAAI/Emu3-Chat-hf`，采用 LoRA-only 微调、selective action mask、两阶段规划等核心设计。

## 当前状态

- 主干模型：`BAAI/Emu3-Chat-hf`
- 微调方式：LoRA only，冻结底模参数
- 低显存路径：支持 4-bit 量化加载
- 权威主链路：`scripts/` + `unitok_drive_lite/`
- 数据：支持合成 toy dataset 与最小 `nuScenes` 适配器

## 实现边界

当前 unified-token 的实现方式如下：

- **文本 + 图像**：由 Emu3 的 `AutoProcessor + chat template` 处理，文本导航信息与前视图 / 当前 BEV 图像作为多模态输入送入主干
- **action / future BEV**：先通过离散化编码为固定全局 special token（`<UT_ACT_XXXX>` / `<UT_BEV_XXXX>`），再接入共享语言 token 空间
- **历史动作摘要**：作为 summary token 区间（`<UT_SUM_XXXX>`）接入
- **训练监督**：只监督 future action token 和 future BEV token

也就是说，共享空间的核心逻辑是：

1. Emu3 负责文本和当前图像的上下文建模
2. 动作与未来 BEV 通过固定离散 token 接入同一 LM 词表
3. 通过 selective attention mask 控制 future action token 的可见性

这不是"把所有模态都编码成同一种原生 token"，而是一个更务实的混合方案。

## Selective Action Mask

项目实现了 selective attention mask，核心规则：

- future action token 之间不能互相看见
- 但它们可以看到：
  - 文本 token
  - 当前前视图图像 token
  - 当前 BEV token
  - 历史动作 summary token

当前分两种路径：

- **训练 / 全序列前向**：使用 4D additive mask，精确保留 selective 约束
- **`generate(...)` 路径**：如果底层 `transformers` / Emu3 对 4D mask 不稳定，会自动回退到 2D padding mask

需要注意：selective mask 在训练路径更准确；在 `generate(...)` 路径可能退化为近似实现，尚未充分验证稳定性。因此严格的 selective 推理更推荐使用 planner 的 "query 打分 / rollout" 方式。

## Token 设计

项目使用全局固定 token registry，所有 special token 在初始化时一次性注册到 tokenizer：

- 结构化标签 token：`<BEV>` `<ACT>` `<PLAN>` `<DREAM>` `<ACT_SUMMARY>` `<NAV_LEFT>` `<NAV_RIGHT>` 等
- action token：`<UT_ACT_0000>` ~ `<UT_ACT_XXXX>`，固定全局区间
- future BEV token：`<UT_BEV_0000>` ~ `<UT_BEV_XXXX>`，固定全局区间
- history action summary token：`<UT_SUM_0000>` ~ `<UT_SUM_XXXX>`，固定全局区间

同一个训练集、同一个推理流程、同一个 tokenizer 扩词顺序下，token id 保持一致。

当前主链路的词表策略是：

- **使用 added special tokens + `resize_token_embeddings(...)`**
- **不复用 Emu3 现有普通语义 token id**

唯一的 token source of truth 在：

- `unitok_drive_lite/token_registry.py`

当前实现会在模型初始化时显式校验：

- tokenizer 词表大小
- 输入 embedding 词表大小
- lm_head / logits 词表大小
- registry 解析出的 token id 是否唯一且全部位于模型词表范围内

因此，如果本地 `hf_models/Emu3-Chat-hf` 目录里的 tokenizer 已被扩词，而模型权重词表还停留在旧大小，主链路会自动做 resize 并在训练前打印调试信息。

## 项目边界

这是一个最小原型，不是完整自动驾驶栈，也不是可直接部署的系统。当前边界：

- `UnifiedDrivingSample` 是当前最小主链路的统一样本格式
- `ToyUnifiedDriveDataset` 是合成 toy 数据，只用于跑通主流程
- `NuScenesUnifiedDriveDataset` 是最小适配器，不是 benchmark 级官方评测管线
- planner 的 scorer 默认是启发式规则，不是 learned scorer
- future BEV 离散化是最小实现，不是高保真世界模型
- selective mask 在 `generate(...)` 路径可能退化为近似实现
- 顶层 `train/`、`infer/`、`models/`、`data/` 为早期实验目录，不再作为主链路维护

## 权威代码路径

当前建议把下面这条路径视为唯一主链路：

- `scripts/train_minimal.py` — 最小训练闭环
- `scripts/run_demo.py` — 最小推理演示
- `unitok_drive_lite/` — 核心包（config / data / model / masking / discretizer / token_registry / train_utils）

后续迭代优先修改这套代码。

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

如果要使用 `nuScenes` 最小适配器，还需要额外安装官方 devkit：

```bash
pip install nuscenes-devkit
```

### 低显存说明

- Linux + CUDA 环境下可以使用 `bitsandbytes` 做 4-bit 量化
- CPU 或非 CUDA 环境也可以运行 toy 逻辑，但不适合实际大模型训练

## 运行方式

### 1. 最小训练闭环

使用包内 dataset factory，跑通 `dataset -> discretizer -> model -> train_utils` 全链路：

```bash
python3 scripts/train_minimal.py --dataset_type toy --dataset_size 8 --num_epochs 1
```

可选参数：

- `--dataset_type`：数据源类型，支持 `toy` / `nuscenes`
- `--dataset_size`：toy 数据集样本数（默认 8）
- `--nuscenes_root`：nuScenes 数据根目录，仅 `dataset_type=nuscenes` 时必填
- `--nuscenes_version`：nuScenes 版本，默认 `v1.0-mini`
- `--nuscenes_split`：nuScenes split，默认 `mini_train`
- `--max_samples`：限制 nuScenes 样本数，便于最小 smoke test
- `--num_epochs`：训练轮数（默认 1）
- `--output_dir`：checkpoint 保存路径（默认 `outputs/unitok_drive_lite`）

训练完成后 checkpoint 保存到：

```
outputs/unitok_drive_lite/checkpoint_last
```

最小 `nuScenes` 训练示例：

```bash
python3 scripts/train_minimal.py \
  --dataset_type nuscenes \
  --nuscenes_root /path/to/nuscenes \
  --nuscenes_version v1.0-mini \
  --nuscenes_split mini_train \
  --max_samples 2 \
  --num_epochs 1
```

### 2. 最小推理演示

```bash
python3 scripts/run_demo.py \
  --dataset_type toy \
  --checkpoint_dir outputs/unitok_drive_lite/checkpoint_last
```

可选参数：

- `--checkpoint_dir`：checkpoint 路径（默认 `outputs/unitok_drive_lite/checkpoint_last`）
- `--dataset_type`：数据源类型，支持 `toy` / `nuscenes`
- `--dataset_size`：toy demo 数据集大小，默认 `8`
- `--nuscenes_root`：nuScenes 数据根目录，仅 `dataset_type=nuscenes` 时必填
- `--nuscenes_version`：nuScenes 版本，默认 `v1.0-mini`
- `--nuscenes_split`：nuScenes split，默认 `mini_train`
- `--max_samples`：限制 nuScenes demo 可索引样本数
- `--sample_index`：选用第几个样本（默认 0）

最小 `nuScenes` demo 示例：

```bash
python3 scripts/run_demo.py \
  --dataset_type nuscenes \
  --checkpoint_dir outputs/unitok_drive_lite/checkpoint_last \
  --nuscenes_root /path/to/nuscenes \
  --nuscenes_version v1.0-mini \
  --nuscenes_split mini_train \
  --max_samples 2 \
  --sample_index 0
```

### 3. 两阶段 planner mock demo

不依赖真实 Emu3 权重，可直接跑通 "动作候选 -> future BEV rollout -> 打分 -> 轨迹解码"：

```bash
python3 infer/planner.py
```

### 4. 自定义 SFT 训练脚本

如果你已经有自己的样本文件，可以使用顶层实验目录中的手写训练循环版本：

```bash
python3 train/train_sft.py --data_path path/to/your_samples.pt
```

注意：此脚本属于早期实验路径，不是当前主链路。

## 关键实现说明

### Emu3 输入组织

- 当前图像上下文交给 `AutoProcessor` 处理
- future action / future BEV 作为离散 token 追加到目标序列

### LoRA 与新增 token

- 默认冻结底模
- 只训练 LoRA 参数（rank=16, alpha=32）
- 以及新增 special token 对应的 embedding / lm_head 行

当前主链路不会把整块 `embed_tokens` / `lm_head` 设为可训练后再做梯度掩码，
而是只为“新增 token 行”保留小规模参数，这样更适合 24G 显存级别的 LoRA 微调。

### 两阶段 planner

planner 拆成四个核心函数：

- `generate_action_candidates(...)` — 采样 K 个候选动作序列
- `rollout_future_bev(...)` — 对每个候选预测未来 BEV
- `score_candidates(...)` — 启发式打分（risk / progress / comfort）
- `plan_once(...)` — 完整规划入口

默认 scorer 是启发式实现，已预留自定义 learned scorer 接口。
