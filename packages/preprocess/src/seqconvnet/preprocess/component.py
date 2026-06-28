"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from seqconvnet.core import VoxelParameters
from virid.core import component
from dataclasses import dataclass


@component()
@dataclass()
class PreprocessConfig:
    train_las_folder: str
    test_las_folder: str
    preprocessed_folder: str
    area_size: float
    voxel_params: VoxelParameters
    device: str = "cpu"


@component()
@dataclass()
class PreprocessInfo:
    mapping: dict[int, int]
    num_classes: int 