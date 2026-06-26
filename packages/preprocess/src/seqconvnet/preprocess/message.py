"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from dataclasses import dataclass, field
from typing import Optional
from virid.core import EventMessage
from seqconvnet.core import VoxelParameters


class TimerMessage(EventMessage): ...


@dataclass
class StartUpMessage(EventMessage):
    train_las_folder: str
    test_las_folder: str
    preprocessed_folder: str
    area_size: float
    voxel_params: VoxelParameters = field(default_factory=VoxelParameters)
    # 额外配置
    delete_labels: Optional[list[int]] = None
    device: str = "cpu"


class MappingMessage(TimerMessage): ...


@dataclass
class DeleteLabelsMessage(TimerMessage):
    las_folder: str
    delete_labels: list[int] = field(default_factory=list)


@dataclass
class PreprocessMessage(TimerMessage):
    las_folder: str
    overlap: bool


@dataclass
class LinkTestDataMessage(TimerMessage):
    las_folder: str
