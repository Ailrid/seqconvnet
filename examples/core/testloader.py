"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

# 测试测试集加载器是否能正常工作

import torch
from seqconvnet.core import TestLoader, VoxelParameters
from torch.utils.data import DataLoader


def test_test_loader():

    device = "cuda"

    voxel_params = VoxelParameters(
        xy_resolution=0.5,
        z_resolution=0.5,
        max_z=64,
        min_rows=128,
        min_cols=128,
    )
    loader = TestLoader(
        root_folder="preprocessed/dales_las/test",
        voxel_params=voxel_params,
        device=device,
    )
    loader = DataLoader(loader, batch_size=1)
    # 统计label总数
    unique_labels_set = set()
    for input_mat, valid_len_mat, label_mat in loader:
        current_labels = torch.unique(label_mat)
        unique_labels_set.update(current_labels.cpu().numpy())

    print(unique_labels_set)


test_test_loader()
