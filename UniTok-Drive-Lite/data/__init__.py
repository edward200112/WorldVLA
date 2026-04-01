"""Deprecated: 顶层 data 实验组件集合。

当前仓库的唯一权威运行主链路是 `scripts/` + `unitok_drive_lite/`。
这里保留的数据工具只用于实验和兼容旧调用，不再作为官方主链路的一部分维护。
"""

from .bev_rasterizer import BEVRasterizer, BEVRasterizerConfig, render_bev

__all__ = ["BEVRasterizer", "BEVRasterizerConfig", "render_bev"]
