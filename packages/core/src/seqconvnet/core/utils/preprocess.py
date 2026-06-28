"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from tqdm import tqdm
import laspy
import os
import numpy as np
import torch
from ..structs import LabelMat, LasPoints, DataMat, VoxelParameters


def read_las_file(path: str) -> LasPoints:
    """
    读取LAS文件
    """

    las_file = laspy.read(path)

    return LasPoints(
        torch.tensor(las_file.xyz),
        torch.from_numpy(np.array(las_file.classification, dtype=np.int32)),
        path,
    )


def read_las_fold(path: str):
    """
    读取LAS文件夹
    """
    file_name = [file for file in os.listdir(path) if file.endswith(".las")]
    for file in file_name:
        yield read_las_file(os.path.join(path, file))


def generate_label(
    data: LasPoints,
    data_mat: DataMat,
    num_classes: int,
    device: str,
) -> LabelMat:
    """生成对应的las块的标签矩阵"""
    num_classes = num_classes + 1
    # 准备基础数据
    voxel_indices = data_mat.full_indices.to(device).long()  # 每个点对应的体素一维索引
    labels = data.classifications.to(device).long()  # 每个点的标签

    num_voxels = data_mat.num_rows * data_mat.num_cols * data_mat.max_z

    # 线性编码映射
    # 将 (voxel_index, label) 二维元组编码为一维索引
    combined_index = voxel_indices * num_classes + labels

    # 向量化计算频次
    # minlength 确保即使后面的体素没有点，也会被分配合适的统计空间
    counts = torch.bincount(combined_index, minlength=num_voxels * num_classes)

    # 重塑回二维 (体素数, 类别数)
    counts = counts.reshape(num_voxels, num_classes)

    # 求 argmax 得到众数
    group_mat = torch.argmax(counts, dim=-1).to(torch.uint8)

    group_mat = group_mat.reshape(
        (data_mat.num_rows, data_mat.num_cols, data_mat.max_z)
    )

    end_padding = torch.zeros(
        (data_mat.num_rows, data_mat.num_cols, 1), dtype=torch.uint8
    ).to(device)
    group_mat = torch.concat((group_mat, end_padding), dim=-1)

    # 获得 label_mat 标签矩阵
    label_mat = torch.take_along_dim(group_mat, data_mat.sort_index, dim=-1)

    # 开头拼上开始 token
    start_mask = torch.full(
        (data_mat.num_rows, data_mat.num_cols, 1),
        num_classes,
        dtype=torch.uint8,
        device=device,
    )
    teach_mat = torch.concat((start_mask, label_mat), dim=-1)

    label_mat = label_mat[:, :, : data_mat.max_z_voxel]
    teach_mat = teach_mat[:, :, : data_mat.max_z_voxel]

    return LabelMat(
        label_mat.permute(2, 0, 1).to(torch.int64),
        teach_mat.permute(2, 0, 1).to(torch.int64),
    )


def generate_data(
    data: LasPoints,
    params: VoxelParameters,
    device: str,
) -> DataMat:
    """
    生成对应的las块的数据矩阵
    """
    points = data.points.to(device)
    xy_resolution = params.xy_resolution
    z_resolution = params.z_resolution
    max_z = params.max_z

    # 计算坐标范围
    max_coord, _ = torch.max(points, dim=0)
    min_coord, _ = torch.min(points, dim=0)
    boundary = (max_coord - min_coord).to("cpu")
    # 如果未指定，自动计算行列数

    num_rows = int(torch.ceil(boundary[0] / xy_resolution).to(torch.int64))
    num_cols = int(torch.ceil(boundary[1] / xy_resolution).to(torch.int64))

    if num_rows < params.min_rows:
        num_rows = params.min_rows
    if num_cols < params.min_cols:
        num_cols = params.min_cols

    # 计算每个点的矩阵索引
    x_indices = torch.floor((points[:, 0] - min_coord[0]) / xy_resolution).to(
        torch.int64
    )
    y_indices = torch.floor((points[:, 1] - min_coord[1]) / xy_resolution).to(
        torch.int64
    )
    z_indices = torch.floor((points[:, 2] - min_coord[2]) / z_resolution).to(
        torch.int64
    )

    x_indices = torch.clamp(x_indices, min=0, max=num_rows - 1)
    y_indices = torch.clamp(y_indices, min=0, max=num_cols - 1)
    z_indices = torch.clamp(z_indices, min=0, max=max_z - 1)

    # 构建一维索引
    full_indices = x_indices * num_cols * max_z + y_indices * max_z + z_indices
    # 对每个有效位置的体素，都置1
    input_mask_mat = torch.zeros(
        (num_rows, num_cols, max_z), dtype=torch.uint8, device=device
    )
    input_mask_mat[x_indices, y_indices, z_indices] = 1
    # 如果没有设定z方向上最大的高程点数,自动计算整个区域内最长的序列长度
    valid_input_len = torch.sum(input_mask_mat, dim=-1)
    max_z_voxel = valid_input_len.max() + 1

    # 在最后一个维度上拼接一个结束占位符
    end_mask = torch.ones((num_rows, num_cols, 1), dtype=torch.uint8, device=device)
    input_end_mask_mat = torch.concat((input_mask_mat, end_mask), dim=-1).to(device)
    # 对矩阵mask排序，以排序获得的下标作为高程
    sort_input, sort_index = torch.sort(
        input_end_mask_mat, dim=-1, descending=True, stable=True
    )
    # sort_input为1代表高程有效,否则得0表示无效高程点，相乘获得点云矩阵形式
    # sort_index+1,把0留给填充点
    input_mat = sort_input * (sort_index + 1)
    # 裁剪到指定长度
    input_mat = input_mat[:, :, :max_z_voxel]
    sort_input = sort_input[:, :, :max_z_voxel]
    padding = torch.zeros((sort_input.shape[0], sort_input.shape[1], 1)).to(device)
    # 丢掉第一个有效长度，从而排除掉结束标签
    sort_input = torch.cat((sort_input[:, :, 1:], padding), dim=-1)
    return DataMat(
        # 按照索引排序的点云序列矩阵
        input_mat.permute(2, 0, 1).to(torch.int64),
        # 沿着高程方向上mask，有效体素为1
        sort_input.permute(2, 0, 1).to(torch.int64),
        x_indices,
        y_indices,
        z_indices,
        full_indices,
        sort_index,
        num_rows,
        num_cols,
        max_z,
        int(max_z_voxel),
    )


def chunk_area(
    data: LasPoints,
    area_size: float,
    overlap: bool,
    device: str,
):
    """
    点云切块迭代器
    """
    points = data.points.to(device)
    classifications = data.classifications.to(device)

    max_coord, _ = torch.max(points, dim=0)
    min_coord, _ = torch.min(points, dim=0)

    stride = area_size / 2.0 if overlap else area_size

    x_range = int((max_coord[0] - min_coord[0]) // stride)
    y_range = int((max_coord[1] - min_coord[1]) // stride)
    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1

    # 计算常规边界
    x_indices = torch.arange(x_range, device=points.device)
    starts_x = min_coord[0] + (x_indices * stride).to(dtype=points.dtype)
    ends_x = starts_x + area_size

    # 强行让最后一块的终点等于最大值，把小尾巴连着吐出去
    ends_x[-1] = max_coord[0]

    y_indices = torch.arange(y_range, device=points.device)
    starts_y = min_coord[1] + (y_indices * stride).to(dtype=points.dtype)
    ends_y = starts_y + area_size
    ends_y[-1] = max_coord[1]

    over_y = ends_y > max_coord[1]
    ends_y[over_y] = max_coord[1]
    starts_y[over_y] = torch.maximum(min_coord[1], ends_y[over_y] - area_size)
    # 对 X 轴进行一次排序
    sort_idx = torch.argsort(points[:, 0])
    points_sorted = points[sort_idx]
    classifications_sorted = classifications[sort_idx]

    # 一次性用二分查找算出所有 X 窗口对应的点云索引区间
    left_indices = torch.searchsorted(points_sorted[:, 0].contiguous(), starts_x)
    right_indices = torch.searchsorted(points_sorted[:, 0].contiguous(), ends_x)

    # 迭代
    for x in range(x_range):
        l_idx = left_indices[x].item()
        r_idx = right_indices[x].item()

        # 如果这个 X 条带内根本没有点，直接跳过整个 Y 循环
        if l_idx == r_idx:
            continue

        # 提取当前 X 条带范围内的点
        stripe_points = points_sorted[l_idx:r_idx]
        stripe_class = classifications_sorted[l_idx:r_idx]
        stripe_orig_idx = sort_idx[l_idx:r_idx]  # 记录这些点在原点云中的真实索引

        for y in range(y_range):
            st_y = starts_y[y]
            ed_y = ends_y[y]

            # 只在当前 X 条带内做 Y 轴过滤
            y_mask = (stripe_points[:, 1] >= st_y) & (stripe_points[:, 1] < ed_y)

            if not y_mask.any():
                continue

            # 提取切块数据
            area_data = stripe_points[y_mask]
            area_classification = stripe_class[y_mask]

            # 这些点在原始点云里的索引
            valid_orig_idx = stripe_orig_idx[y_mask]

            yield (
                valid_orig_idx,
                LasPoints(
                    area_data,
                    area_classification,
                    data.path,
                ),
                x,
                y,
            )


def reindex_label(data: LasPoints, mapping: dict[int, int]):
    """
    标签重映射
    """
    if not mapping:
        return
    max_old_label = max(mapping.keys())
    lut = torch.arange(max_old_label + 1, dtype=torch.int16)
    for old_label, new_label in mapping.items():
        lut[old_label] = new_label
    data.classifications = lut[data.classifications]


def preprocess_las_file(
    las_path: str,
    data_folder: str,
    label_folder: str,
    num_classes: int,
    mapping: dict[int, int],
    area_size: float,
    overlap: bool,
    params: VoxelParameters,
    device: str,
) -> None:
    """预处理单个LAS文件"""
    las_data = read_las_file(las_path)
    file_name = os.path.basename(las_path).split(".")[0]
    # 不存在两个路径就创建
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    # 重新索引标签
    reindex_label(las_data, mapping)
    # 切块生成四个矩阵
    for _, chunk, x, y in chunk_area(las_data, area_size, overlap, device):
        # 获取文件名
        data_mat = generate_data(chunk, params, device)
        label_mat = generate_label(chunk, data_mat, num_classes, device)
        # 写入切块的数据和标签
        # header = laspy.LasHeader(point_format=3, version="1.2")
        # header.offsets = chunk.points.cpu().numpy().mean(axis=0)
        # header.scales = np.array([0.01, 0.01, 0.01])
        # chunk_las = laspy.LasData(header=header)
        # chunk_points = chunk.points.cpu().numpy()
        # chunk_las.x = chunk_points[:, 0]
        # chunk_las.y = chunk_points[:, 1]
        # chunk_las.z = chunk_points[:, 2]
        # chunk_las.classification = chunk.classifications.cpu().numpy().astype(np.uint8)
        # chunk_las.write(os.path.join(data_folder, f"{file_name}_{x}_{y}.las"))
        torch.save(
            data_mat.input_mat.cpu(),
            os.path.join(data_folder, f"{file_name}_{x}_{y}.input"),
        )
        torch.save(
            data_mat.valid_len_mat.cpu(),
            os.path.join(data_folder, f"{file_name}_{x}_{y}.valid_len"),
        )
        # 写入标签
        torch.save(
            label_mat.label_mat.cpu(),
            os.path.join(label_folder, f"{file_name}_{x}_{y}.label"),
        )
        torch.save(
            label_mat.teach_mat.cpu(),
            os.path.join(label_folder, f"{file_name}_{x}_{y}.teach"),
        )


def mae_preprocess_las_file(
    las_path: str,
    data_folder: str,
    mapping: dict[int, int],
    area_size: float,
    overlap: bool,
    params: VoxelParameters,
    device: str,
) -> None:
    """mae_预处理单个LAS文件"""
    las_data = read_las_file(las_path)
    file_name = os.path.basename(las_path).split(".")[0]
    # 不存在两个路径就创建
    os.makedirs(data_folder, exist_ok=True)
    # 重新索引标签
    reindex_label(las_data, mapping)
    # 切块生成四个矩阵
    for _, chunk, x, y in chunk_area(las_data, area_size, overlap, device):
        # 获取文件名
        data_mat = generate_data(chunk, params, device)
        torch.save(
            data_mat.input_mat.cpu(),
            os.path.join(data_folder, f"{file_name}_{x}_{y}.input"),
        )
        torch.save(
            data_mat.valid_len_mat.cpu(),
            os.path.join(data_folder, f"{file_name}_{x}_{y}.valid_len"),
        )


def delete_labels(las_path: str, labels: list[int]):
    """
    删除指定标签内的点，并直接覆盖存储原文件
    """
    las_data = laspy.read(las_path)
    mask = ~np.isin(np.array(las_data.classification), labels)
    filtered_points = las_data.points[mask]
    new_las = laspy.LasData(las_data.header)
    new_las.points = filtered_points  # type: ignore
    new_las.write(las_path)


def label_map(las_folder: str) -> tuple[dict[int, int], int, torch.Tensor]:
    """
    生成标签映射使标签连续: old_label -> new_label
    完全基于 PyTorch 进行标签统计、类别推导与权重计算。
    """
    # 用于全局累加各个标签的数量
    global_label_counts = {}
    files = [f for f in os.listdir(las_folder) if f.endswith(".las")]

    with tqdm(files, desc="Mapping & Weighting (PyTorch)", leave=False) as pbar:
        for file_name in pbar:
            las_path = os.path.join(las_folder, file_name)

            with laspy.open(las_path) as fh:
                las_data = fh.read()
                # 读出数据直接丢给 PyTorch Tensor，告别 NumPy
                current_labels = torch.tensor(
                    las_data.classification, dtype=torch.int64
                )

                # 使用 torch.unique 进行去重和计数
                unique_labels, counts = torch.unique(current_labels, return_counts=True)

                # 累加到全局字典中
                for label, count in zip(unique_labels, counts):
                    label_id = int(label.item())
                    global_label_counts[label_id] = (
                        global_label_counts.get(label_id, 0) + count.item()
                    )

    # 排序并生成连续标签映射 (old_label -> new_label)
    unique_labels = sorted(list(global_label_counts.keys()))
    mapping = {
        old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)
    }

    num_unique_labels = len(unique_labels)
    num_classes = num_unique_labels  # 自动推导类别数

    # 权重计算
    counts_tensor = torch.tensor(
        [global_label_counts[label] for label in unique_labels], dtype=torch.float32
    )

    label_weights = counts_tensor + 0.001
    label_weights /= torch.sum(label_weights)
    label_weights = torch.pow(torch.max(label_weights) / label_weights, 1.0 / 3.0)

    # 映射到最终的 weights_tensor 中
    weights_tensor = torch.ones(num_classes, dtype=torch.float32)

    for old_label, label_w in zip(unique_labels, label_weights):
        new_label = mapping[old_label]
        weights_tensor[new_label - 1] = label_w

    return mapping, num_unique_labels, weights_tensor
