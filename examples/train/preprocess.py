"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from virid.core import create_virid
from virid.std import StdPlugin
from seqconvnet.core import VoxelParameters
from seqconvnet.preprocess.component import bind_components
from seqconvnet.preprocess.message import StartUpMessage
from seqconvnet.preprocess.hook import activate_hook
from seqconvnet.preprocess.system import register_systems

virid = create_virid().use(StdPlugin, None)

activate_hook(virid)
bind_components(virid)
register_systems(virid)
# 启动
StartUpMessage.send(
    preprocess_target="training",
    train_las_folder="dales_las/train",
    test_las_folder="dales_las/test",
    preprocessed_folder="preprocessed",
    # 默认训练参数
    voxel_params=VoxelParameters(
        xy_resolution=0.5,
        z_resolution=0.5,
        max_z=128,
        min_rows=128,
        min_cols=128,
    ),
    area_size=128,
    # 额外配置
    delete_labels=None,
    device="cuda",
)
virid.tick()

# 仅用于可视化测试
# from seqconvnet.utils.pred import get_label_from_mat


# def restore_mat(
#     las_path: str,
#     label_mat_path: str,
#     mat_point_path: str,
#     restore_point_path: str,
#     voxel_params: VoxelParameters,
# ) -> None:
#     """
#     尝试利用序列矩阵恢复一个点云来看看
#     """
#     raw_data = read_las_file(las_path)
#     data_mat = generate_data(raw_data, voxel_params, "cpu")
#     label_mat: Tensor3D = torch.load(label_mat_path).to("cpu")
#     # 手动生成一个点云
#     # 写入切块的数据和标签
#     header = laspy.LasHeader(point_format=3, version="1.2")
#     header.offsets = np.array([0.0, 0.0, 0.0])
#     header.scales = np.array([0.01, 0.01, 0.01])
#     las_data = laspy.LasData(header=header)

#     valid_len_mat = data_mat.valid_len_mat
#     input_mat = data_mat.input_mat
#     nz_indices = torch.nonzero(valid_len_mat)
#     k_idx, i_idx, j_idx = nz_indices[:, 0], nz_indices[:, 1], nz_indices[:, 2]

#     # 向量化计算坐标
#     x_coords = i_idx.float() * voxel_params.xy_resolution
#     y_coords = j_idx.float() * voxel_params.xy_resolution
#     z_coords = input_mat[k_idx, i_idx, j_idx].float() * voxel_params.z_resolution
#     labels = label_mat[k_idx, i_idx, j_idx]

#     # 直接赋值给 laspy
#     las_data.x = x_coords.numpy()
#     las_data.y = y_coords.numpy()
#     las_data.z = z_coords.numpy()
#     las_data.classification = labels.numpy().astype(np.uint8)
#     las_data.write(mat_point_path)
#     ######################################################################################
#     # 用反排序索引恢复高程的位置
#     # 拿到真正的每个点的结果
#     pred_point_label = get_label_from_mat(label_mat, data_mat, "cpu")
#     header = laspy.LasHeader(point_format=3, version="1.2")
#     header.offsets = raw_data.points.numpy().mean(axis=0)
#     header.scales = np.array([0.01, 0.01, 0.01])
#     las_data = laspy.LasData(header=header)
#     las_data.x = raw_data.points.numpy()[:, 0]
#     las_data.y = raw_data.points.numpy()[:, 1]
#     las_data.z = raw_data.points.numpy()[:, 2]
#     las_data.classification = pred_point_label.cpu().numpy().astype(np.uint8)
#     las_data.write(restore_point_path)


# restore_mat(
#     las_path="preprocessed/dales_las/train/data/5080_54435.las_0_0.las",
#     label_mat_path="preprocessed/dales_las/train/label/5080_54435.las_0_0.label",
#     mat_point_path="mat_point.las",
#     restore_point_path="restore_point_path.las",
#     voxel_params=VoxelParameters(
#         xy_resolution=0.5,
#         z_resolution=0.5,
#         max_z=64,
#         num_rows=None,
#         num_cols=None,
#     ),
# )
