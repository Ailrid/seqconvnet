"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from ..structs import DataMat, LabelMat, Tensor3D
import torch
import random
from typing import List


def rand_flip(mats: List[Tensor3D]) -> List[Tensor3D]:
    """同步翻转一组矩阵 (形状均为 (dim, H, W))"""
    if random.random() > 0.5:
        mats = [torch.flip(m, dims=[-1]) for m in mats]  # 左右翻转
    if random.random() > 0.5:
        mats = [torch.flip(m, dims=[-2]) for m in mats]  # 上下翻转
    return mats


def rand_rotate(mats: List[Tensor3D]) -> List[Tensor3D]:
    """同步直角旋转一组矩阵 (0/90/180/270度，无插值污染)"""
    k = random.randint(0, 3)
    if k > 0:
        mats = [torch.rot90(m, k=k, dims=[-2, -1]) for m in mats]
    return mats


def rand_crop(mats: List[Tensor3D], req_h: int, req_w: int) -> List[Tensor3D]:
    """从一组矩阵中同步随机裁剪出指定的宽高 (req_h, req_w)"""
    H, W = mats[0].shape[-2], mats[0].shape[-1]
    start_row = random.randint(0, H - req_h)
    start_col = random.randint(0, W - req_w)
    return [
        m[:, start_row : start_row + req_h, start_col : start_col + req_w] for m in mats
    ]


def montage(
    area_1: Tensor3D,
    area_2: Tensor3D,
    area_3: Tensor3D,
    area_4: Tensor3D,
    x_pos: int,
    y_pos: int,
    input_size: int,
    max_dim: int,
) -> Tensor3D:
    """
    将 4 个不同尺寸、不同 dim 的局部块拼成最终 (max_dim, input_size, input_size) 的矩阵。
    短的 dim 在高维通道处自动留 0 占位。
    """
    # 初始化全 0 大画布
    output = torch.zeros(
        (max_dim, input_size, input_size), dtype=area_1.dtype, device=area_1.device
    )

    output[: area_1.shape[0], :x_pos, :y_pos] = area_1  # 左上 (来自样本A)
    output[: area_2.shape[0], :x_pos, y_pos:] = area_2  # 右上 (来自样本B)
    output[: area_3.shape[0], x_pos:, :y_pos] = area_3  # 左下 (来自样本B)
    output[: area_4.shape[0], x_pos:, y_pos:] = area_4  # 右下 (来自样本A)

    return output


def enhance(
    input_mat: Tensor3D,
    valid_len_mat: Tensor3D,
    label_mat: Tensor3D,
    teach_mat: Tensor3D,
    other_input_mat: Tensor3D,
    other_valid_len_mat: Tensor3D,
    other_label_mat: Tensor3D,
    other_teach_mat: Tensor3D,
    input_size: int,
):
    # 组装两个独立样本的组件群
    group_A = [
        input_mat,
        valid_len_mat,
        label_mat,
        teach_mat,
    ]
    group_B = [
        other_input_mat,
        other_valid_len_mat,
        other_label_mat,
        other_teach_mat,
    ]

    # 计算合并画布所需的最大 dim
    dim_A = group_A[0].shape[0]
    dim_B = group_B[0].shape[0]
    max_dim = max(dim_A, dim_B)

    # 旋转与翻转增强
    group_A = rand_flip(rand_rotate(group_A))
    group_B = rand_flip(rand_rotate(group_B))

    # 随机决定最终 input_size 画布内部的十字分割点
    x_pos = int(torch.randint(input_size // 4, 3 * input_size // 4, (1,)).item())
    y_pos = int(torch.randint(input_size // 4, 3 * input_size // 4, (1,)).item())

    # 计算 4 个象限各自所需的空间高宽
    h1, w1 = x_pos, y_pos
    h2, w2 = x_pos, input_size - y_pos
    h3, w3 = input_size - x_pos, y_pos
    h4, w4 = input_size - x_pos, input_size - y_pos

    # 分别裁剪 4 个象限
    # 每个 area 都是一个包含 4 个组件的 List，且保持各自组的 dim (dim_A 或 dim_B)
    area_1 = rand_crop(group_A, h1, w1)  # 左上 -> 用 A
    area_2 = rand_crop(group_B, h2, w2)  # 右上 -> 用 B
    area_3 = rand_crop(group_B, h3, w3)  # 左下 -> 用 B
    area_4 = rand_crop(group_A, h4, w4)  # 右下 -> 用 A

    # 把 4 个组件缝合起来
    final_input = montage(
        area_1[0], area_2[0], area_3[0], area_4[0], x_pos, y_pos, input_size, max_dim
    )
    final_valid = montage(
        area_1[1], area_2[1], area_3[1], area_4[1], x_pos, y_pos, input_size, max_dim
    )
    final_label = montage(
        area_1[2], area_2[2], area_3[2], area_4[2], x_pos, y_pos, input_size, max_dim
    )

    final_teach = montage(
        area_1[3], area_2[3], area_3[3], area_4[3], x_pos, y_pos, input_size, max_dim
    )

    # 长度截断逻辑
    l = int(torch.sum(final_valid, dim=0).max().item())
    if l == 0:
        l = 1

    # 组装并返回
    return (
        final_input[:l, :, :],
        final_valid[:l, :, :],
        final_label[:l, :, :],
        final_teach[:l, :, :],
    )


def get_las_mat(
    input_mat: Tensor3D,
    valid_len_mat: Tensor3D,
    label_mat: Tensor3D,
    teach_mat: Tensor3D,
    input_size: int,
):
    _, num_rows, num_cols = input_mat.shape

    start_row = torch.randint(0, num_rows - input_size, (1,))[0]
    start_col = torch.randint(0, num_cols - input_size, (1,))[0]

    area_input_mat = input_mat[
        :, start_row : start_row + input_size, start_col : start_col + input_size
    ]
    area_valid_len_mat = valid_len_mat[
        :, start_row : start_row + input_size, start_col : start_col + input_size
    ]
    area_label_mat = label_mat[
        :, start_row : start_row + input_size, start_col : start_col + input_size
    ]
    area_teach_mat = teach_mat[
        :, start_row : start_row + input_size, start_col : start_col + input_size
    ]

    return area_input_mat, area_valid_len_mat, area_label_mat, area_teach_mat

