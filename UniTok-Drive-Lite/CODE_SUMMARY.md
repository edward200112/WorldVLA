# UniTok-Drive-Lite 代码总结

## 1. 项目目标

这个最小版原型围绕 unified-token 自动驾驶建模展开，核心目标是把以下信息放进同一个 token 空间中：

- 文本导航信息
- 前视图图像
- 当前 BEV 栅格图
- 历史动作摘要
- 未来动作 token
- 未来 3 帧 BEV token

主干模型使用 Hugging Face 的 `facebook/chameleon-7b`，训练方式默认是：

- 冻结底模参数
- 只训练 LoRA
- 只训练新增 special token 对应的 embedding / lm_head 行

---

## 2. 这轮新增的主要文件

### 2.1 `models/action_tokenizer.py`

作用：

- 把未来 1 秒轨迹 `actions: [B, T, 3]` 离散成 action token
- 再从离散 token 重建连续轨迹

结构：

- encoder: MLP
- codebook: 可学习 embedding
- quantize: 最近邻量化
- decoder: 从离散 code 重建轨迹

核心接口：

- `encode_to_indices(actions) -> indices`
- `decode_from_indices(indices) -> recon_actions`
- `forward(actions) -> {"recon", "indices", "vq_loss", "recon_loss", "loss"}`

特点：

- 是一个最小版 VQ-VAE 风格轨迹离散器
- 可以独立训练
- 文件末尾有最小 smoke check

---

### 2.2 `data/bev_rasterizer.py`

作用：

- 把自动驾驶场景渲染成 RGB BEV 栅格图

输入支持：

- `ego_history`
- `vehicle_centers`
- `pedestrian_centers`
- `lane_polylines`
- `drivable_polygon`

输出：

- `PIL.Image.Image`
- 默认大小 `224x224`

颜色约定：

- ego 历史轨迹：白色
- 车辆：蓝色
- 行人：红色
- 车道线：绿色
- 可行驶区域：深灰色

特点：

- ego 位于图像中心
- 前方朝上
- 不依赖 GIS 库
- 主要使用 `numpy + PIL`
- 文件末尾有一个最小 demo，会生成 `data/bev_demo.png`

---

### 2.3 `models/attention_mask.py`

作用：

- 为 unified-token 自动驾驶模型构造 selective attention mask

token 类型枚举：

- `TEXT`
- `IMAGE`
- `CURRENT_BEV`
- `HISTORY_ACTION_SUMMARY`
- `FUTURE_ACTION`
- `FUTURE_BEV`
- `PAD`

核心规则：

- 基本规则是 causal mask
- 对 `FUTURE_ACTION` token 做特殊限制
- future action query 只能看到：
  - 文本 token
  - 当前图像 token
  - 当前 BEV token
  - 历史动作 summary token
  - 自己当前位置对角线

也就是说：

- 它不能看到本 chunk 中更早的 raw action token
- 也不会看到其他未来 token

核心接口：

- `build_selective_attention_mask(token_types) -> [1, 1, L, L]`
- `print_attention_mask_visualization(...)`

---

### 2.4 `models/backbone_chameleon.py`

作用：

- 加载 `facebook/chameleon-7b`
- 加载 `ChameleonProcessor`
- 挂接 LoRA
- 处理 special tokens
- 提供统一前向入口

核心接口：

- `build_model_and_processor(...)`
- `add_special_tokens(...)`
- `forward_batch(...)`

默认 special tokens：

- `<BEV>`
- `<ACT>`
- `<PLAN>`
- `<DREAM>`
- `<ACT_SUMMARY>`
- `<NAV_LEFT>`
- `<NAV_RIGHT>`
- `<NAV_STRAIGHT>`
- `<LIGHT_RED>`
- `<LIGHT_GREEN>`
- `<RISK_PED>`
- `<RISK_VEH>`
- `<RISK_OCC>`

支持能力：

- 4-bit 量化加载
- LoRA 微调
- 预留自动驾驶 special tokens
- 默认冻结底模
- 仅训练 LoRA + 新增 token 行

实现细节：

- 新增 token embedding 不是整块放开训练
- 而是通过 gradient hook 只保留新词表行的梯度

---

### 2.5 `train/train_sft.py`

作用：

- 实现 unified-token 自动驾驶最小版 SFT 训练脚本

特点：

- 不依赖 Hugging Face Trainer
- 使用手写 PyTorch 训练循环
- 支持梯度累计
- 支持 AMP
- 支持 checkpoint 保存
- 每步打印 4 个损失

batch 至少支持的字段：

- `text_prompt`
- `front_image`
- `bev_image`
- `future_action_indices`
- `future_bev_tokens` 或 `future_bev_image`

如果要计算重建损失，还需要：

- `future_actions`

总损失：

```text
loss = loss_action + lambda_bev * loss_bev + lambda_recon * loss_recon
```

3 项损失含义：

- `loss_action`: future action token 交叉熵
- `loss_bev`: future BEV token 交叉熵
- `loss_recon`: action tokenizer 的轨迹重建损失

脚本内部做了这些事：

1. 加载 Chameleon backbone + LoRA
2. 构造 action / BEV 动态 token 词表
3. 把 batch 组装成 unified-token 序列
4. 生成 selective attention mask
5. 分别计算 action / BEV / recon loss
6. 做梯度累计与 optimizer step
7. 定期保存 checkpoint

---

### 2.6 `infer/planner.py`

作用：

- 实现 unified-token 自动驾驶最小版推理器

核心接口：

- `plan_once(...)`

输入：

- 文本导航信息
- 前视图图像
- 当前 BEV 图像
- 历史动作 token，可选

推理流程：

1. backbone 生成 `K` 个候选未来 action token 序列
2. 对每个动作候选，再预测未来 3 帧 BEV token
3. 对每个候选计算简单分数：
   - `risk_score`
   - `progress_score`
   - `comfort_score`
   - `final_score`
4. 选出 `final_score` 最高的候选
5. 用 `ActionTokenizer.decode_from_indices()` 解码回连续轨迹

评分逻辑是启发式的：

- `risk_score`: 基于预测 future BEV token 的占用强度
- `progress_score`: 基于最终前向位移
- `comfort_score`: 基于轨迹平滑程度

文件末尾有一个 mock demo，不依赖真实 Chameleon 权重也能跑通。

---

## 3. 现有代码之间的关系

### 3.1 训练路径

训练阶段的最小链路可以概括为：

1. `ActionTokenizer`
   - 把连续未来轨迹编码为 action token
   - 并提供轨迹重建损失

2. `train_sft.py`
   - 读取 batch
   - 构造 unified-token 序列
   - 生成 `token_types`
   - 用 `attention_mask.py` 构造 selective mask

3. `backbone_chameleon.py`
   - 加载 `ChameleonProcessor + ChameleonForConditionalGeneration`
   - 注入 special tokens
   - 应用 LoRA

4. `train_sft.py`
   - 计算
     - action token loss
     - future BEV token loss
     - action recon loss
   - 汇总为总损失并反向传播

---

### 3.2 推理路径

推理阶段的最小链路可以概括为：

1. 输入：
   - 导航文本
   - 前视图图像
   - 当前 BEV 图像
   - 历史动作 token

2. `planner.py`
   - 采样 K 个 action token 候选
   - 为每个候选生成 future BEV token
   - 计算评分

3. `ActionTokenizer`
   - 把最佳 action token 解码成连续轨迹

4. 输出：
   - 最佳候选
   - 候选分数
   - 连续轨迹

---

## 4. 目前项目的整体结构

当前 `UniTok-Drive-Lite` 目录的核心结构如下：

```text
UniTok-Drive-Lite/
├── CODE_SUMMARY.md
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── bev_demo.png
│   └── bev_rasterizer.py
├── infer/
│   ├── __init__.py
│   └── planner.py
├── models/
│   ├── __init__.py
│   ├── action_tokenizer.py
│   ├── attention_mask.py
│   └── backbone_chameleon.py
├── train/
│   └── train_sft.py
├── scripts/
│   ├── run_demo.py
│   └── train_minimal.py
└── unitok_drive_lite/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── discretizer.py
    ├── masking.py
    ├── model.py
    └── train_utils.py
```

---

## 5. 当前实现的边界

这个版本是“最小可运行原型”，重点是把主流程搭起来，因此有以下边界：

### 已具备

- action tokenizer
- BEV rasterizer
- selective attention mask
- Chameleon + LoRA backbone 封装
- 手写 SFT 训练脚本
- 最小版 planner 推理器

### 仍然是最小实现

- future BEV tokenizer 目前是简单 patch 平均量化
- planner scorer 是启发式打分，不是 learned value head
- `train_sft.py` 目前依赖用户提供 `.pt` 格式样本列表
- 真实 Chameleon 权重的端到端训练，还需要本地装齐依赖并登录 Hugging Face

---

## 6. 当前已做过的验证

已完成的检查包括：

- 所有新增 Python 文件都通过了 `compileall`
- `action_tokenizer.py` 做过前向 + 反向 smoke test
- `bev_rasterizer.py` 的 demo 能生成 `bev_demo.png`
- `attention_mask.py` 已验证 future action 间的可见性规则
- `planner.py` 的 mock demo 可以成功生成候选并输出最佳轨迹

尚未完成的检查：

- 真实 `facebook/chameleon-7b` 的加载和端到端训练
- 真实多模态 driving dataset 的实跑

原因是当前本机环境还缺少：

- `transformers`
- `peft`
- `bitsandbytes`
- Hugging Face gated 模型访问权限 / 登录状态

---

## 7. 下一步建议

如果要把这个最小版原型真正推进到“能训练真实数据”的阶段，建议按下面顺序继续：

1. 安装依赖并验证 `facebook/chameleon-7b` 能正常加载
2. 定义统一的训练样本 schema
3. 写一个真实数据集读取器，把导航 / 前视图 / BEV / future trajectory 对齐
4. 把 `planner.py` 接到真实 checkpoint 上做闭环推理
5. 视需要补充 learned scorer 或 value head

---

## 8. 一句话总结

当前这套代码已经把 unified-token 自动驾驶最小版的关键积木补齐了：

- 连续轨迹可离散成 action token
- 场景可栅格化成 BEV 图
- 模型可使用 selective attention mask
- 主干可加载 Chameleon + LoRA
- 可做最小版训练
- 可做最小版推理规划

它现在已经是一个结构清晰、可继续接真实数据和真实模型权重的原型底座。
