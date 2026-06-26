"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
from seqconvnet.core import (
    TrainLoader,
    TestLoader,
    VoxelParameters,
    Network,
    SegmentationEvaluator,
    SoftDiceAndFocalLoss,
)
from virid.core import component, ViridApp
from dataclasses import dataclass
from torch.utils.data import DataLoader


@component()
@dataclass()
class ModelConfig:
    model: Network = None  # type: ignore
    loss: SoftDiceAndFocalLoss = None  # type: ignore


@component()
@dataclass()
class DatasetConfig:
    batch_size: int = 1
    input_size: int = 128
    train_loader: DataLoader[TrainLoader] = None  # type: ignore
    test_loader: DataLoader[TestLoader] = None  # type: ignore
    voxel_params: VoxelParameters = None  # type: ignore
    num_classes: int = 0


@component()
@dataclass()
class EnvConfig:
    lr: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5
    evaluator: SegmentationEvaluator = None  # type: ignore
    optimizer: torch.optim.Optimizer = None  # type: ignore
    scheduler: torch.optim.lr_scheduler.SequentialLR = None  # type: ignore
    device: str = "cpu"


def bind_initialization_components(app: ViridApp):
    app.bind(ModelConfig)
    app.bind(DatasetConfig)
    app.bind(EnvConfig)
