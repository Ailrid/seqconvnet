"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

# 测试训练加载器是否能正常工作

from seqconvnet.core import TrainLoader, Tensor3D, VoxelParameters
import laspy
import torch
import numpy as np
from torch.utils.data import DataLoader


def restore_mat(
    input_mat: Tensor3D,
    label_mat: Tensor3D,
    valid_len_mat: Tensor3D,
    mat_point_path: str,
    voxel_params: VoxelParameters,
) -> None:
    # 手动生成一个点云
    input_mat = input_mat.to("cpu")
    label_mat = label_mat.to("cpu")
    valid_len_mat = valid_len_mat.to("cpu")

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.scales = np.array([0.01, 0.01, 0.01])
    las_data = laspy.LasData(header=header)

    nz_indices = torch.nonzero(valid_len_mat)
    k_idx, i_idx, j_idx = nz_indices[:, 0], nz_indices[:, 1], nz_indices[:, 2]

    # 向量化计算坐标
    x_coords = i_idx.float() * voxel_params.xy_resolution
    y_coords = j_idx.float() * voxel_params.xy_resolution
    z_coords = input_mat[k_idx, i_idx, j_idx].float() * voxel_params.z_resolution
    labels = label_mat[k_idx, i_idx, j_idx]

    # 直接赋值给 laspy
    las_data.x = x_coords.numpy()
    las_data.y = y_coords.numpy()
    las_data.z = z_coords.numpy()
    las_data.classification = labels.numpy().astype(np.uint8)
    las_data.write(mat_point_path)


def test_train_loader():

    device = "cuda"
    mat_point_path = "mat_point.las"
    voxel_params = VoxelParameters(
        xy_resolution=0.5,
        z_resolution=0.5,
        max_z=64,
        min_rows=128,
        min_cols=128,
    )
    loader = TrainLoader(
        root_folder="preprocessed/dales_las/train",
        iter_times=1,
        input_size=128,
        voxel_params=voxel_params,
    )
    loader = DataLoader(loader, batch_size=1)

    # 随机生成一次，保存结果
    loader_iter = iter(loader)
    input_mat, label_mat, valid_len_mat, teach_mat = next(loader_iter)

    restore_mat(
        input_mat[0],
        label_mat[0],
        valid_len_mat[0],
        mat_point_path,
        voxel_params,
    )
    print(
        "Saved successfully, please open mat_point.las and observe if the category of the point cloud is correct"
    )
    print(input_mat[0, :, 64, 64])
    print(label_mat[0, :, 64, 64])
    print(valid_len_mat[0, :, 64, 64])
    print(teach_mat[0, :, 64, 64])


test_train_loader()
