"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceAndFocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        classes_weights: list[float],
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 注册为 buffer，跟随 self.to(device) 自动移动
        self.register_buffer(
            "classes_weights", torch.tensor(classes_weights, dtype=torch.float32)
        )

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # pred 形状: [B, C, S, H, W] -> 对应的网络输出
        # label 形状: [B, S, H, W]   -> 对应的原始标签 (1 ~ C, 0为背景)
        C = self.num_classes

        # 维度调整与扁平化：[B, C, ...] -> [B, ..., C] -> [N, C]
        pred_filtered = pred.movedim(1, -1).reshape(-1, C).contiguous()
        # 标签扁平化: [B, ...] -> [N]
        target_filtered = label.reshape(-1).contiguous()

        # 过滤背景
        legal_mask = target_filtered != 0
        pred_filtered = pred_filtered[legal_mask]
        target_filtered = target_filtered[legal_mask]

        # 防御性编程：如果当前整个 batch 全是背景，直接返回 0，并保持梯度链完整
        if target_filtered.size(0) == 0:
            return pred.sum() * 0

        # 从 1 ~ C 映射到 0 ~ C-1，完美对齐 pred 维度的 0 ~ C-1
        target_filtered = target_filtered - 1

        log_pt = F.log_softmax(pred_filtered, dim=-1)
        log_pt = log_pt.gather(1, target_filtered.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        # 根据每个像素的真实类别，取出对应的全局类别权重
        batch_weights = self.classes_weights[target_filtered]  # type: ignore
        focal_loss = -batch_weights * ((1 - pt) ** self.gamma) * log_pt

        # 用当前 Batch 内所有有效像素的权重总和进行归一化
        focal_loss = focal_loss.sum() / (batch_weights.sum() + 1e-6)

        # ================================================================= #
        # 5. 动态类别加权 Multiclass Dice Loss
        # ================================================================= #
        pred_prob = F.softmax(pred_filtered, dim=-1)
        target_one_hot = F.one_hot(
            target_filtered, num_classes=self.num_classes
        ).float()

        # 计算每个类别在当前 batch 中的交集和并集，形状均为 [C]
        intersection = torch.sum(pred_prob * target_one_hot, dim=0)
        cardinality = torch.sum(pred_prob + target_one_hot, dim=0)

        # 每一类的标准 Dice Loss (加入 1.0 平滑项防止极值震荡)
        dice_loss_per_class = 1 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

        # 动态找出当前 Batch 中真正存在（至少出现过一次）的类别
        # target_one_hot.sum(dim=0) 的形状是 [C]，代表每个类在当前 batch 里的像素总数
        class_exist_mask = target_one_hot.sum(dim=0) > 0  # 得到一个 Bool 掩码 [C]

        # 不存在的类别权重直接归零，存在的类别保留其全局配置权重
        valid_weights = self.classes_weights * class_exist_mask  # type: ignore

        # 只对存在的类别进行动态归一化求和
        dice_loss = torch.sum(dice_loss_per_class * valid_weights) / (
            valid_weights.sum() + 1e-6
        )

        return self.alpha * focal_loss + self.beta * dice_loss


class MaeVoxelLoss(nn.Module):
    def __init__(
        self,
        max_z: int = 128,  # 预训练时的 num_classes 就是最大高度层数
        gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = max_z
        self.gamma = gamma

    def forward(
        self, pred: torch.Tensor, label: torch.Tensor, mae_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            pred:     模型的网络预测输出 [B, num_classes(128), S, H, W]
            label:    没有被污染的原始高度序列 [B, S, H, W]
            mae_mask: Shell 返回的空间掩码矩阵 [B, S, H, W] (True 表示被擦除需要计算 Loss)
        """
        # 调整维度 [B, num_classes, S, H, W] -> [B, S, H, W, num_classes]
        pred_permuted = pred.permute(0, 2, 3, 4, 1).contiguous()
        mask_bool = mae_mask.bool()

        # 只切出真正被 MAE 挖掉的那些稀疏体素
        pred_filtered = pred_permuted[mask_bool]  # [N, 128]
        target_filtered = label[mask_bool].long()  # [N]

        # 将原始高度 1~128 统一减 1，映射到 0~127 索引
        target_filtered = target_filtered - 1

        # 边界过滤
        legal_mask = (target_filtered >= 0) & (target_filtered < self.num_classes)
        pred_filtered = pred_filtered[legal_mask]
        target_filtered = target_filtered[legal_mask]

        if target_filtered.size(0) == 0:
            return pred.sum() * 0

        # Focal Loss
        log_pt = F.log_softmax(pred_filtered, dim=-1)
        log_pt = log_pt.gather(1, target_filtered.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_loss = -((1 - pt) ** self.gamma) * log_pt

        return focal_loss.mean()


Loss = Union[
    SoftDiceAndFocalLoss,
    MaeVoxelLoss,
]
