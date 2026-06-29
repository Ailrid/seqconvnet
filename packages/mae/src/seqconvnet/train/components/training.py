"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from virid.core import component, ViridApp
from dataclasses import dataclass, field
from seqconvnet.core import SegmentationMetrics


@component()
@dataclass()
class TrainingState:
    # 此次训练的开始时间
    timestamp: str = ""
    # 当前的轮数
    current_epoch: int = 0
    # 当前epoch的损失
    current_metrics: float = 0
    # 最好的一次损失
    best_metrics: float = 0
    # 日志路径
    log_folder: str = ""
    # 模型保存路径
    checkpoint_folder: str = ""


def bind_training_components(app: ViridApp):
    app.bind(TrainingState)
