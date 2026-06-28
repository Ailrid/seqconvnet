"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from seqconvnet.core import (
    generate_data,
    generate_label,
    read_las_file,
    VoxelParameters,
)

las_path = "dales_las/train/5080_54435.las"
device = "cuda"
num_classes = 8
force_teach_token = num_classes + 2
voxel_params = VoxelParameters(
    xy_resolution=0.5,
    z_resolution=0.5,
    max_z=256,
    min_rows=128,
    min_cols=128,
)


data = read_las_file(las_path)
data_mat = generate_data(data, voxel_params, device)
label = generate_label(data, data_mat, num_classes, device)
print(data_mat.input_mat.shape)
print(data_mat.valid_len_mat.shape)
print(label.label_mat.shape)
print(label.teach_mat.shape)

print(data_mat.input_mat[:, 20, 20])
print(label.label_mat[:, 20, 20])
print(data_mat.valid_len_mat[:, 20, 20])
print(label.teach_mat[:, 20, 20])
