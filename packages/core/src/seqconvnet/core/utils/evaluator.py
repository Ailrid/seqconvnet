"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from typing import Optional, overload

import torch
from ..structs import Tensor4D, Tensor2D, Tensor1D, SegmentationMetrics


def compute_hist_4d(
    prediction: Tensor4D,  # [B, S, H, W]
    label: Tensor4D,  # [B, S, H, W]
    valid_mask: Tensor4D,  # [B, S, H, W]
    num_classes: int,
) -> Tensor2D:
    """
    计算当前 batch 的混淆矩阵
    """
    bool_mask = valid_mask.bool()

    # 提取有效一维向量（现在它们会安全地展平成一维）
    pred_valid = prediction[bool_mask].long()
    label_valid = label[bool_mask].long()

    # 将 1~num_classes 的标签平移到 0~(num_classes-1)
    label_shifted = label_valid - 1
    pred_shifted = pred_valid - 1

    # 同时对 label 和 pred 进行合法性检查
    # 确保 label 和 pred 平移后都在 0 ~ num_classes-1 的合法范围内
    legal_mask = (
        (label_shifted >= 0)
        & (label_shifted < num_classes)
        & (pred_shifted >= 0)
        & (pred_shifted < num_classes)
    )

    # 用共同的 legal_mask 提取最终用于计算的像素
    final_label = label_shifted[legal_mask]
    final_pred = pred_shifted[legal_mask]

    # 计算混淆矩阵
    hist = (
        torch.bincount(num_classes * final_label + final_pred, minlength=num_classes**2)
        .reshape(num_classes, num_classes)
        .float()
    )

    return hist


def compute_hist_1d(
    prediction: Tensor1D,  # [N] 预测的一维向量
    label: Tensor1D,  # [N] 真实标签的一维向量
    num_classes: int,
) -> Tensor2D:
    """
    计算一维预测和标签向量的混淆矩阵
    """

    # 转换为 long 类型以进行索引计算
    pred_valid = prediction.long()
    label_valid = label.long()

    # 将 1 ~ num_classes 的标签平移到 0 ~ (num_classes - 1)
    label_shifted = label_valid - 1
    pred_shifted = pred_valid - 1

    # 对 label 和 pred 进行合法性检查，剔除可能存在的越界或占位符
    legal_mask = (
        (label_shifted >= 0)
        & (label_shifted < num_classes)
        & (pred_shifted >= 0)
        & (pred_shifted < num_classes)
    )

    # 提取最终用于计算的有效元素
    final_label = label_shifted[legal_mask]
    final_pred = pred_shifted[legal_mask]

    # 利用 bincount 高效计算混淆矩阵
    hist = (
        torch.bincount(num_classes * final_label + final_pred, minlength=num_classes**2)
        .reshape(num_classes, num_classes)
        .float()
    )

    return hist


class SegmentationEvaluator:
    def __init__(self, num_classes: int, voxel_model: bool):
        self.num_classes = num_classes
        self.total_hist = None
        self.voxel_model = voxel_model

    def reset(self):
        """每个 Epoch 开始前调用，清空累计的混淆矩阵"""
        self.total_hist = None

    @overload
    def update(self, prediction: Tensor1D, label: Tensor1D): ...

    @overload
    def update(self, prediction: Tensor4D, label: Tensor4D, valid_mask: Tensor4D): ...

    def update(
        self,
        prediction: Tensor4D | Tensor1D,
        label: Tensor4D | Tensor1D,
        valid_mask: Optional[Tensor4D] = None,
    ):
        """每个 Batch 结束时调用，累计混淆矩阵"""
        # 计算当前 batch 的混淆矩阵
        if self.voxel_model and valid_mask is not None:
            batch_hist = compute_hist_4d(
                prediction, label, valid_mask, self.num_classes
            )
        else:
            batch_hist = compute_hist_1d(prediction, label, self.num_classes)

        # 累加到全局混淆矩阵中
        if self.total_hist is None:
            self.total_hist = torch.zeros_like(batch_hist)
        self.total_hist += batch_hist

    def compute_metrics(self) -> SegmentationMetrics:
        """Epoch 结束时调用，计算最终的所有分割指标"""
        if self.total_hist is None:
            raise RuntimeError("Confusion matrix is empty. Please call update() first.")

        hist = self.total_hist
        diag = torch.diag(hist)  # 各类别预测正确的数量 (TP)
        row_sum = hist.sum(dim=1)  # 各类别真实的像素总数 (TP + FN)
        col_sum = hist.sum(dim=0)  # 各类别预测的像素总数 (TP + FP)

        # 计算每一类的 Recall (召回率) / Acc (准确率) -> 分母是真实值
        # Recall = TP / (TP + FN)
        recall_per_class = diag / (row_sum + 1e-6)

        # 计算每一类的 Precision (精确率) -> 分母是预测值
        # Precision = TP / (TP + FP)
        precision_per_class = diag / (col_sum + 1e-6)

        # 计算每一类的 IoU
        # IoU = TP / (TP + FN + FP)
        iou_per_class = diag / (row_sum + col_sum - diag + 1e-6)

        # 计算全局平均指标
        mIoU = torch.nanmean(iou_per_class)
        mRecall = torch.nanmean(recall_per_class)
        mPrecision = torch.nanmean(precision_per_class)

        return SegmentationMetrics(
            mIoU=mIoU.item(),
            mRecall=mRecall.item(),
            mPrecision=mPrecision.item(),
            IoU=iou_per_class.tolist(),
            Recall=recall_per_class.tolist(),
            Precision=precision_per_class.tolist(),
        )

    def print_metrics(self) -> tuple[list[list[int]], SegmentationMetrics]:
        """美观地在终端中打印分割指标表格与混淆矩阵"""
        metrics: SegmentationMetrics = self.compute_metrics()

        # =====================================================================
        # 打印分割核心指标表 (IoU, Recall, Precision)
        # =====================================================================
        print("\n" + "=" * 66)
        print(" SEGMENTATION EVALUATION REPORT ".center(66, "="))
        print("=" * 66)

        # 内部列宽分配：Class ID(14), IoU(14), Recall(14), Precision(16)
        top_line = (
            "┌" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 16 + "┐"
        )
        mid_divider = (
            "├" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 16 + "┤"
        )
        bottom_line = (
            "└" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 16 + "┘"
        )

        print(top_line)
        print(
            f"│ {'Class ID':^12} │ {'IoU (%)':^12} │ {'Recall (%)':^12} │ {'Precision (%)':^14} │"
        )
        print(mid_divider)

        for i in range(self.num_classes):
            iou_val = metrics.IoU[i] * 100
            rec_val = metrics.Recall[i] * 100
            prec_val = metrics.Precision[i] * 100

            print(
                f"│ {f'Class {i+1}':<12} │ "
                f"{iou_val:^12.2f} │ "
                f"{rec_val:^12.2f} │ "
                f"{prec_val:^14.2f} │"
            )

        print(mid_divider)

        print(
            f"│ {'Mean (mAvg)':<12} │ "
            f"\033[1;32m{metrics.mIoU*100:^12.2f}\033[0m │ "
            f"{metrics.mRecall*100:^12.2f} │ "
            f"{metrics.mPrecision*100:^14.2f} │"
        )
        print(bottom_line)
        # =====================================================================
        # 动态绘制混淆矩阵表 (Confusion Matrix - Percentage Normalized by Row)
        # =====================================================================
        print("\n" + "=" * 78)
        print(" CONFUSION MATRIX (Row: GT, Col: Pred) [% Normalized] ".center(78, "="))
        print("=" * 78)

        if self.total_hist is None:
            print(" [Warning] Confusion matrix is empty. No data accumulated yet.")
            print("=" * 78 + "\n")
            raise RuntimeError("Confusion matrix is empty. Please call update() first.")

        # 使用 float 类型的矩阵进行比例计算
        hist_matrix = self.total_hist.float().tolist()

        # 定义矩阵单元格的图表宽度 (百分比字符串包含%, 稍微加宽到 12 保证排版美观)
        head_w = 14  # 第一列（GT行名）的宽度
        cell_w = 14  # 后续每个数据单元格（Pred比例）的宽度

        # 动态根据类别数构建混淆矩阵的边框组件
        cm_top = "┌" + "─" * head_w + ("┬" + "─" * cell_w) * self.num_classes + "┐"
        cm_mid = "├" + "─" * head_w + ("┼" + "─" * cell_w) * self.num_classes + "┤"
        cm_bottom = "└" + "─" * head_w + ("┴" + "─" * cell_w) * self.num_classes + "┘"

        # 打印混淆矩阵表头 (Prediction 横轴)
        print(cm_top)
        header_row = f"│ {'GT Pred':^{head_w-2}} │"
        for j in range(self.num_classes):
            header_row += f" {f'Pred {j + 1}':^{cell_w-2}} │"
        print(header_row)
        print(cm_mid)

        # 逐行打印混淆矩阵实体数据 (Ground Truth 纵轴)
        for i in range(self.num_classes):
            data_row = f"│ {f'Class {i+1}':^{head_w-2}} │"

            # 计算当前行 (GT Class i) 的样本总数，用于按行归一化
            row_total = sum(hist_matrix[i])

            for j in range(self.num_classes):
                count_val = hist_matrix[i][j]

                # 计算百分比字符串，如果该类总数为0则展示 0.00%
                if row_total > 0:
                    percentage_str = f"{(count_val / row_total) * 100:.2f} %"
                else:
                    percentage_str = "0.00 %"

                # 特色功能：如果是对角线上的元素（预测正确 TP / Recall），加粗高亮显示
                if i == j:
                    # 亮绿色或青色加粗显示核心指标 Recall
                    data_row += f" \033[1;32m{percentage_str:^{cell_w-2}}\033[0m │"
                else:
                    # 非对角线（错分率）如果是 0.00% 保持清爽，如果有值可以用普通文本
                    if percentage_str == "0.00 %":
                        data_row += f" \033[90m{percentage_str:^{cell_w-2}}\033[0m │"  # 灰色暗显0%，突出错误
                    else:
                        data_row += f" {percentage_str:^{cell_w-2}} │"
            print(data_row)

        print(cm_bottom)
        print("=" * 78 + "\n")

        return hist_matrix, metrics
