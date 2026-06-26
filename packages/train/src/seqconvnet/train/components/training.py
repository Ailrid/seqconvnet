"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from virid.core import component, ViridApp
from dataclasses import dataclass


@component()
@dataclass()
class TrainingState:
    # 此次训练的开始时间
    timestamp: str = ""
    # 当前的轮数
    current_epoch: int = 0


def bind_training_components(app: ViridApp):
    app.bind(TrainingState)
