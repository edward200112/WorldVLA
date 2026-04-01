"""Deprecated: 顶层 infer 实验组件集合。

当前仓库的唯一权威运行主链路是 `scripts/` + `unitok_drive_lite/`。
这里导出的 planner 仅保留给实验和兼容旧调用，不再作为官方推理入口维护。
"""

from .planner import generate_action_candidates, plan_once, rollout_future_bev, score_candidates

__all__ = [
    "generate_action_candidates",
    "rollout_future_bev",
    "score_candidates",
    "plan_once",
]
