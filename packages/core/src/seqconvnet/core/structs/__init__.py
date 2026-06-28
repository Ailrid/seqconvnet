"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from dataclasses import dataclass, field
import torch
from typing import Annotated, TypeAlias

Tensor5D: TypeAlias = Annotated[torch.Tensor, "Shape: (B, D, S, R, C)"]
Tensor4D: TypeAlias = Annotated[torch.Tensor, "Shape: (B, M, R, C)"]
Tensor3D: TypeAlias = Annotated[torch.Tensor, "Shape: (B, D, S)"]
Tensor2D: TypeAlias = Annotated[torch.Tensor, "Shape: (B, D)"]
Tensor1D: TypeAlias = Annotated[torch.Tensor, "Shape: (B,)"]


@dataclass
class LasPoints:
    # n*3的float32矩阵
    points: Tensor2D
    # n*1的int32矩阵
    classifications: Tensor2D
    # las文件路径
    path: str

@dataclass
class DataMat:
    """
    数据矩阵
    """

    # 输入矩阵和可用长度矩阵
    input_mat: Tensor3D
    valid_len_mat: Tensor3D
    # 其他需要保留的中间信息
    x_indices: Tensor1D
    y_indices: Tensor1D
    z_indices: Tensor1D
    full_indices: Tensor1D
    sort_index: Tensor3D
    num_rows: int
    num_cols: int
    max_z: int
    max_z_voxel: int


@dataclass
class LabelMat:
    """
    标签矩阵
    """

    label_mat: Tensor3D
    teach_mat: Tensor3D


@dataclass
class VoxelParameters:
    """
    数据集大小参数
    """

    xy_resolution: float
    z_resolution: float
    max_z: int
    min_rows: int
    min_cols: int


@dataclass
class SegmentationMetrics:
    """
    语义分割指标
    """

    mIoU: float = 0.0
    mRecall: float = 0.0  # 平均召回率
    mPrecision: float = 0.0  # 平均精确率
    IoU: list[float] = field(default_factory=list)
    Recall: list[float] = field(default_factory=list)
    Precision: list[float] = field(default_factory=list)
