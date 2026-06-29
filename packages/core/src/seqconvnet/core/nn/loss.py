"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module): ...


class SoftDiceAndFocalLoss(Loss):  # 确保继承自 nn.Module
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

        # 注册为 buffer，这样 self.to(device) 时它会自动移动，不需要手动管
        self.register_buffer(
            "classes_weights", torch.tensor(classes_weights, dtype=torch.float32)
        )

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        #  pred: [B, C, S, H, W], label: [B, S, H, W]
        C = self.num_classes

        # 将通道/类别维移到最后: [B, C, ...] -> [B, ..., C] -> [N, C]
        pred_filtered = pred.movedim(1, -1).reshape(-1, C).contiguous()
        # 标签展平: [B, ...] -> [N]
        target_filtered = label.reshape(-1).contiguous()

        # 过滤背景
        legal_mask = target_filtered != 0
        pred_filtered = pred_filtered[legal_mask]
        target_filtered = target_filtered[legal_mask]

        # 如果这个 batch 全是背景，直接返回 0
        if target_filtered.size(0) == 0:
            return pred.sum() * 0

        # 从 1~num_classes 映射到 0~num_classes-1
        target_filtered = target_filtered - 1

        # 加权 Focal Loss
        log_pt = F.log_softmax(pred_filtered, dim=-1)
        log_pt = log_pt.gather(1, target_filtered.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        batch_weights = self.classes_weights[target_filtered]  # type: ignore
        focal_loss = -batch_weights * ((1 - pt) ** self.gamma) * log_pt

        focal_loss = focal_loss.sum() / (batch_weights.sum() + 1e-6)

        # ================================================================= #
        # 4. 加权 Multiclass Dice Loss
        # ================================================================= #
        pred_prob = F.softmax(pred_filtered, dim=-1)
        target_one_hot = F.one_hot(
            target_filtered, num_classes=self.num_classes
        ).float()

        # 计算每个类别在当前 batch 中的交集和并集
        intersection = torch.sum(pred_prob * target_one_hot, dim=0)
        cardinality = torch.sum(pred_prob + target_one_hot, dim=0)

        # 核心改进：添加平滑项，同时避免未出现类别贡献脏梯度
        # 分子分母同时加 1 可以有效缓解“未出现类别”的 Loss 异常波动
        dice_loss_per_class = 1 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

        # 进阶建议（可选）：如果你希望当前 batch 没出现的类别完全不参与 Dice 计算，可以解开下方注释：
        class_exist_mask = target_one_hot.sum(dim=0) > 0
        valid_weights = self.classes_weights * class_exist_mask  # type: ignore
        dice_loss = torch.sum(dice_loss_per_class * valid_weights) / (
            valid_weights.sum() + 1e-6
        )

        dice_loss = torch.sum(dice_loss_per_class * self.classes_weights) / (  # type: ignore
            self.classes_weights.sum() + 1e-6  # type: ignore
        )

        return self.alpha * focal_loss + self.beta * dice_loss


class SoftDiceAndCELoss(Loss):  # 规范继承 nn.Module

    def __init__(
        self,
        num_classes: int,
        classes_weights: list[float],
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

        # 推荐：使用 register_buffer，权重会自动随模型 .to(device) 转移，无需手动指定 device
        self.register_buffer(
            "classes_weights", torch.tensor(classes_weights, dtype=torch.float32)
        )

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 支持任意空间维度，不管是 2D [B, C, H, W] 还是 3D [B, C, S, H, W]
        C = self.num_classes

        # 1. 安全降维：将 Channel 维（dim=1）安全地挪到最后，其余维度展平
        pred_filtered = pred.movedim(1, -1).reshape(-1, C).contiguous()
        target_filtered = label.reshape(-1).contiguous()

        # 2. 过滤有效标签（假设 0 是需要过滤的背景或忽略标签）
        legal_mask = target_filtered != 0
        pred_filtered = pred_filtered[legal_mask]
        target_filtered = target_filtered[legal_mask]

        # 如果整张图/当前 batch 全是背景，直接返回 0 梯度
        if target_filtered.size(0) == 0:
            return pred.sum() * 0

        # 先过滤，后对齐（从 1~num_classes 映射到 0~num_classes-1）
        target_filtered = target_filtered - 1

        # ================================================================= #
        # 3. 加权 交叉熵 Loss
        # ================================================================= #
        # 此时 target_filtered 的范围是 0 ~ C-1，完美匹配 classes_weights
        ce_loss = F.cross_entropy(
            pred_filtered,
            target_filtered,
            weight=self.classes_weights,  # type: ignore
            reduction="mean",
        )

        # ================================================================= #
        # 4. 加权 Multiclass Dice Loss
        # ================================================================= #
        pred_prob = F.softmax(pred_filtered, dim=-1)
        target_one_hot = F.one_hot(target_filtered, num_classes=C).float()

        # 统计当前 batch 中每个类别的交集和并集
        intersection = torch.sum(pred_prob * target_one_hot, dim=0)
        cardinality = torch.sum(pred_prob + target_one_hot, dim=0)

        # 核心修正：使用拉普拉斯平滑（+1.0），防止没出现的类别刷出极大的错误梯度
        dice_loss_per_class = 1 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

        # 如果你希望当前 batch 没出现的类别完全不贡献 Dice 梯度，可以解开下面三行的注释：
        class_exist_mask = target_one_hot.sum(dim=0) > 0
        valid_weights = self.classes_weights * class_exist_mask  # type: ignore
        dice_loss = torch.sum(dice_loss_per_class * valid_weights) / (
            valid_weights.sum() + 1e-6
        )

        dice_loss = torch.sum(dice_loss_per_class * self.classes_weights) / (  # type: ignore
            self.classes_weights.sum() + 1e-6  # type: ignore
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
