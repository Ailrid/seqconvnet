"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from typing import Optional

from seqconvnet.core import VoxelParameters
from dataclasses import dataclass, field


@dataclass()
class ModelParameters:
    checkpoint_folder: Optional[str] = None
    mae_checkpoint_folder: Optional[str] = None
    model_type: str = "transformer"
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1


@dataclass()
class DatasetParameters:
    train_las_folder: str = ""
    test_las_folder: str = ""
    # 从1开始
    num_classes: int = 8
    classes_weights: list[float] = field(default_factory=list)
    classes_names: Optional[list[str]] = None
    batch_size: int = 1
    num_workers: int = 1
    iter_times: int = 1
    input_size: int = 128
    area_size: float = 128
    voxel_params: VoxelParameters = None  # type: ignore


@dataclass()
class EnvParameters:
    lr: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    device: str = "cpu"
