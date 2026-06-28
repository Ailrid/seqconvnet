"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

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
        device: str = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.classes_weights = torch.tensor(classes_weights, dtype=torch.float32).to(
            device
        )

    def forward(
        self, pred: torch.Tensor, label: torch.Tensor, valid_len_mat: torch.Tensor
    ) -> torch.Tensor:
        # [B, num_classes, S, H, W] -> [B, S, H, W, num_classes]
        pred_permuted = pred.permute(0, 2, 3, 4, 1).contiguous()
        mask_bool = valid_len_mat.bool()

        pred_filtered = pred_permuted[mask_bool]  # [N, num_classes]
        target_filtered = label[mask_bool].long()  # [N]

        # 统一减 1 对齐到 0 ~ num_classes-1，刚好和你的 weights_tensor 索引完全重合！
        target_filtered = target_filtered - 1

        legal_mask = (target_filtered >= 0) & (target_filtered < self.num_classes)
        pred_filtered = pred_filtered[legal_mask]
        target_filtered = target_filtered[legal_mask]

        if target_filtered.size(0) == 0:
            return pred.sum() * 0

        # 加权 Focal Loss
        log_pt = F.log_softmax(pred_filtered, dim=-1)
        log_pt = log_pt.gather(1, target_filtered.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        # 此时 target_filtered 的值（0~7）直接作为 self.classes_weights 的索引，绝对不会越界
        batch_weights = self.classes_weights[target_filtered]
        focal_loss = -batch_weights * ((1 - pt) ** self.gamma) * log_pt
        focal_loss = focal_loss.sum() / (batch_weights.sum() + 1e-6)

        # 加权 Multiclass Dice Loss
        pred_prob = F.softmax(pred_filtered, dim=-1)
        target_one_hot = F.one_hot(
            target_filtered, num_classes=self.num_classes
        ).float()

        intersection = torch.sum(pred_prob * target_one_hot, dim=0)
        cardinality = torch.sum(pred_prob + target_one_hot, dim=0)

        dice_loss_per_class = 1 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        # 对应类别乘对应权重，加权平均
        dice_loss = torch.sum(dice_loss_per_class * self.classes_weights) / (
            self.classes_weights.sum() + 1e-6
        )

        return self.alpha * focal_loss + self.beta * dice_loss


class SoftDiceAndCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        classes_weights: list[float],
        alpha: float = 1.0,
        beta: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

        self.classes_weights = torch.tensor(classes_weights, dtype=torch.float32).to(
            device
        )

    def forward(
        self, pred: torch.Tensor, label: torch.Tensor, valid_len_mat: torch.Tensor
    ) -> torch.Tensor:
        pred_permuted = pred.permute(0, 2, 3, 4, 1).contiguous()
        mask_bool = valid_len_mat.bool()

        pred_filtered = pred_permuted[mask_bool]
        target_filtered = label[mask_bool].long()

        # 统一减 1 对齐到 0 ~ num_classes-1
        target_filtered = target_filtered - 1

        legal_mask = (target_filtered >= 0) & (target_filtered < self.num_classes)
        pred_filtered = pred_filtered[legal_mask]
        target_filtered = target_filtered[legal_mask]

        if target_filtered.size(0) == 0:
            return pred.sum() * 0

        # 加权 交叉熵 Loss
        # 这里的 self.classes_weights 长度刚好是 num_classes
        ce_loss = F.cross_entropy(
            pred_filtered,
            target_filtered,
            weight=self.classes_weights,
            reduction="mean",
        )

        # 加权 Multiclass Dice Loss
        pred_prob = F.softmax(pred_filtered, dim=-1)
        target_one_hot = F.one_hot(
            target_filtered, num_classes=self.num_classes
        ).float()

        intersection = torch.sum(pred_prob * target_one_hot, dim=0)
        cardinality = torch.sum(pred_prob + target_one_hot, dim=0)

        dice_loss_per_class = 1 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        dice_loss = torch.sum(dice_loss_per_class * self.classes_weights) / (
            self.classes_weights.sum() + 1e-6
        )

        return self.alpha * ce_loss + self.beta * dice_loss


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
