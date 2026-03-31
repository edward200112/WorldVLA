## 语言增强 waypoint planner

输入：多相机图像序列 + ego 历史状态 + route command + 一段语言输入
输出：未来 5 秒 waypoints + 一个离散行为标签 + 一句简短解释


## 核心任务
主任务：预测未来 5 秒 N=20 个 waypoints（4Hz）
辅任务 1：预测行为标签，例如 keep_lane / stop / yield / lane_change_left / lane_change_right / turn_left / turn_right
辅任务 2：生成一句解释，例如“前方行人进入横道，因此减速保持直行”

## 数据说明


### 主数据：使用Waymo E2E数据集
8 个方向相机图像
10Hz 图像序列
离散 route command：left / straight / right
过去 4 秒 ego 轨迹（4Hz）
ego 速度和加速度
训练集中的未来 5 秒轨迹
某些关键帧还带 rater feedback，对 3 条 5 秒轨迹打分 0–10，用来表示“多种合理驾驶”的偏好信息。


### 语言增强 语言增强：DriveLM
DriveLM 的价值不在于替代主数据，而在于补上主数据里弱的部分：
它把驾驶拆成 Perception、Prediction、Planning、Behavior、Motion 五层，并用人工编写的 reasoning logic 把这些层串起来，还同时提供 nuScenes 和 CARLA 版本。

### 指令接地：Talk2Car
Talk2Car 是建立在 nuScenes 之上的自然语言命令数据，核心是把乘客命令和前视图里的目标物体对齐

### 闭环评测：CARLA + Bench2Drive
CARLA 负责仿真，Bench2Drive 负责系统化 closed-loop benchmark。Bench2Drive 还给了一个小验证集 Dev10，专门用于快速开发和消融。


## 数据格式
x = {
  images: 最近 T 秒多相机图像,
  ego_hist: 最近 4 秒 ego trajectory + speed + accel,
  route_cmd: {left, straight, right},
  lang: 一段文本，可为空
}

y = {
  waypoints: 未来 5 秒二维轨迹点,
  behavior: 离散行为标签,
  rationale: 一句解释文本
}
lang：
1.空文本
用于普通 waypoint 训练，让模型不依赖语言也能开
2.解释型 prompt
例如：
Explain the next driving action briefly.
这种用于让模型学会输出理由
3.指令型 prompt
例如：
Turn right after the parked truck and stop near the crosswalk.
这种用于语言约束规划



## 模型结构：四编码器 + 三输出头
1. 视觉：ViT/BEV
2. Ego 状态编码器 
3. 语言：Qwen2.5
4. 融合层：把 V + E + L 丢进一个 fusion transformer。

输出头
1. Waypoint head：未来 20 个点 (x_t, y_t)
2. Behavior head：行为类别 softmax
3. Rationale head：行为解释说明


## Loss设计
L = λ_wp * L_wp
  + λ_cls * L_behavior
  + λ_txt * L_text
  + λ_smooth * L_smooth
  + λ_pref * L_preference

1. 主损失：L1
L_wp = Σ_t w_t * ||p_t - p_t*||_1

2. L_behavior 交叉熵
3. L_text token-level CE
4. L_smooth
5. L_preference： BCE偏好损失



## 训练流程
无语言planner -> 语言监督 -> 偏好学习
