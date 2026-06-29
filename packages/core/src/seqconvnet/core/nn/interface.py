"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from abc import abstractmethod
from ..structs import Tensor4D
import torch
from ..utils import SegmentationMetrics


class Network(torch.nn.Module):
    """
    网络模型
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def refer(self, input_mat: Tensor4D, valid_len_mat: Tensor4D) -> Tensor4D:
        """ """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        加载模型参数
        """
        raise NotImplementedError

    @abstractmethod
    def load_mae_checkpoint(self, path: str) -> None:
        """
        加载模型参数
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str, best_metrics: SegmentationMetrics) -> None:
        """
        加载模型参数
        """
        raise NotImplementedError

    @abstractmethod
    def save_mae_checkpoint(self, path: str, best_metrics: float) -> None:
        """
        加载模型参数
        """
        raise NotImplementedError
