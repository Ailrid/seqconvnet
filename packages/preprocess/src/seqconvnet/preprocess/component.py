"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from seqconvnet.core import VoxelParameters
from virid.core import component, ViridApp
from dataclasses import dataclass, field


@component()
@dataclass()
class PreprocessConfig:
    train_las_folder: str = ""
    test_las_folder: str = ""
    preprocessed_folder: str = "preprocessed"
    area_size: float = 100.0
    voxel_params: VoxelParameters = field(default_factory=VoxelParameters)
    device: str = "cpu"


@component()
@dataclass()
class PreprocessInfo:
    mapping: dict[int, int] = field(default_factory=dict)
    num_classes: int = 0


def bind_components(app: ViridApp):
    app.bind(PreprocessConfig)
    app.bind(PreprocessInfo)
