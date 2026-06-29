"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
from seqconvnet.core import (
    MaeTrainLoader,
    MaeTestLoader,
    VoxelParameters,
    Network,
    MaeVoxelLoss,
)
from virid.core import component
from dataclasses import dataclass
from torch.utils.data import DataLoader
from ..params import ModelParameters, DatasetParameters, EnvParameters


@component()
@dataclass()
class LightParameters:
    dataset_params: DatasetParameters
    model_params: ModelParameters
    env_params: EnvParameters


@component()
@dataclass()
class ModelConfig:
    model: Network
    loss: MaeVoxelLoss


@component()
@dataclass()
class DatasetConfig:
    batch_size: int
    input_size: int
    train_loader: DataLoader[MaeTrainLoader]
    test_loader: DataLoader[MaeTestLoader]
    voxel_params: VoxelParameters
    num_workers: int
    mask_ratio: float


@component()
@dataclass()
class EnvConfig:
    lr: float
    epochs: int
    warmup_epochs: int
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.SequentialLR
    device: str
