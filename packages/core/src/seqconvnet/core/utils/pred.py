"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import math
import torch
from ..structs import Tensor3D, DataMat, Tensor1D, Tensor4D, VoxelParameters
from ..nn.interface import Network
from ..dataloader.supervised import PredLoader
import laspy


def get_label_from_mat(label_mat: Tensor3D, data_mat: DataMat, device: str) -> Tensor1D:
    """从标签矩阵里获得每个点的结果"""

    label_mat = label_mat.permute(1, 2, 0)

    pred_mat = torch.zeros(
        (
            data_mat.num_rows,
            data_mat.num_cols,
            data_mat.max_z + 1,
        ),
        dtype=torch.int64,
        device=device,
    )

    pred_mat[: label_mat.shape[0], : label_mat.shape[1], : label_mat.shape[2]] = (
        label_mat
    )

    inverse_indices = torch.argsort(data_mat.sort_index.to(device))

    total_label_mat = torch.take_along_dim(pred_mat, inverse_indices, dim=-1)

    # 拿到真正的每个点的结果

    pred_point_label = total_label_mat.contiguous()[
        data_mat.x_indices.to(device),
        data_mat.y_indices.to(device),
        data_mat.z_indices.to(device),
    ]

    return pred_point_label


def yield_input_mat(input_mat: Tensor4D, valid_len_mat: Tensor4D, input_size: int):
    """从数据矩阵里获得输入矩阵"""
    # 计算分块的总行列数
    num_rows = input_mat.shape[2]
    num_cols = input_mat.shape[3]
    row_len = math.ceil(num_rows / input_size)
    col_len = math.ceil(num_cols / input_size)

    for i in range(row_len):
        for j in range(col_len):
            row_begin = i * input_size
            col_begin = j * input_size

            row_end = row_begin + input_size
            col_end = col_begin + input_size

            if col_end > num_cols:
                col_end = num_cols
                col_begin = num_cols - input_size

            if row_end > num_rows:
                row_end = num_rows
                row_begin = num_rows - input_size

            valid_input = valid_len_mat[
                :,
                :,
                row_begin:row_end,
                col_begin:col_end,
            ]

            valid_len = torch.sum(valid_input, dim=1).max()
            if valid_len == 0:
                continue

            yield (
                input_mat[
                    :,
                    :valid_len,
                    row_begin:row_end,
                    col_begin:col_end,
                ],
                (row_begin, col_begin),
            )


def refer_mat(
    input_mat: Tensor4D,
    net: Network,
    input_size: int,
) -> Tensor4D:
    """推理一个不定大小的输入矩阵,输出和input_mat的形状相同"""

    batch_size, num_step, num_rows, num_cols = input_mat.shape
    valid_len_mat = (input_mat != 0).to(torch.int32)
    if num_rows < input_size or num_cols < input_size:
        raise ValueError("The input size is too large.")

    pred_mat = torch.zeros(
        (batch_size, num_step, num_rows, num_cols),
        dtype=torch.int64,
        device=input_mat.device,
    )

    for area_input_mat, pos in yield_input_mat(input_mat, valid_len_mat, input_size):
        pred = net.refer(area_input_mat)
        pred_mat[
            :,
            : pred.shape[1],
            pos[0] : pos[0] + input_size,
            pos[1] : pos[1] + input_size,
        ] = pred

    return pred_mat


def refer_file(
    las_path: str,
    save_path: str,
    net: Network,
    input_size: int,
    area_size: float,
    voxel_params: VoxelParameters,
    device: str,
):

    net = net.to(device).eval()

    dataloader = PredLoader(las_path, area_size, voxel_params, device)

    las_data = dataloader.las_data
    idxs = []
    labels = []

    for idx, data_mat, _ in dataloader:

        pred_mat = refer_mat(
            data_mat.input_mat,
            net,
            input_size,
        )

        # 当前这一组的labels
        label = get_label_from_mat(pred_mat.squeeze(0), data_mat, device)

        labels.append(label)
        idxs.append(idx)

    # 构造一个合适长度的新分类
    labels = torch.cat(labels).to(device, dtype=torch.int32)
    idxs = torch.cat(idxs).to(device, dtype=torch.int32)
    new_labels = torch.zeros(las_data.points.shape[0], dtype=torch.int32, device=device)
    new_labels[idxs] = labels

    # 重新保存
    raw_data = laspy.read(las_path)
    raw_data.classification = new_labels.cpu().to(torch.uint8).numpy()
    raw_data.write(save_path)
