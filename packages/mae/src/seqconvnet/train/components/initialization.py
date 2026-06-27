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
from virid.core import component
from dataclasses import dataclass
from torch.utils.data import DataLoader


@component()
@dataclass()
class ModelConfig:
    model: Network
    loss: SoftDiceAndFocalLoss


@component()
@dataclass()
class DatasetConfig:
    batch_size: int
    input_size: int
    train_loader: DataLoader[TrainLoader]
    test_loader: DataLoader[TestLoader]
    voxel_params: VoxelParameters
    num_classes: int
    num_workers: int


@component()
@dataclass()
class EnvConfig:
    lr: float
    epochs: int
    warmup_epochs: int
    evaluator: SegmentationEvaluator
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.SequentialLR
    device: str
