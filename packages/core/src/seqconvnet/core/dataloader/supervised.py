"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import os
import torch
from ..utils.preprocess import generate_data, chunk_area, read_las_file
from ..utils.train import enhance, get_las_mat
from ..structs import VoxelParameters, LabelMat
from torch.utils.data import IterableDataset


class PredLoader(IterableDataset):
    """
    预测阶段的单文件预测数据集
    """

    def __init__(
        self,
        las_path: str,
        area_size: float,
        voxel_params: VoxelParameters,
        device: str,
    ):

        self.las_data = read_las_file(las_path)

        self.area_size = area_size
        self.voxel_params = voxel_params
        self.device = device

    def __iter__(self):
        # 大文件切块
        for idx, chunk, _, _ in chunk_area(
            self.las_data, self.area_size, False, self.device
        ):
            # 对每一块再切块
            data_mat = generate_data(chunk, self.voxel_params, self.device)
            yield idx, data_mat, chunk


class TrainLoader(IterableDataset):
    """
    训练阶段的多文件训练数据集
    """

    def __init__(
        self,
        root_folder: str,
        iter_times: int,
        input_size: int,
        voxel_params: VoxelParameters,
        device: str,
    ):
        self.iter_times = iter_times
        self.data_folder = os.path.join(root_folder, "data")
        self.label_folder = os.path.join(root_folder, "label")
        self.file_list = [f for f in os.listdir(self.data_folder) if f.endswith(".las")]
        self.input_size = input_size
        self.voxel_params = voxel_params
        self.device = device
        self.enhance_key = True

    def toggle_enhance(self):
        self.enhance_key = not self.enhance_key

    def __len__(self):
        return len(self.file_list) * self.iter_times

    def __iter__(self):
        if self.enhance_key:
            for _ in range(len(self.file_list)):
                _, data_mat, label_mat = self.rand_read()
                _, other_data_mat, other_label_mat = self.rand_read()
                # 迭代 iter_times 次
                for _ in range(self.iter_times):
                    yield enhance(
                        data_mat,
                        label_mat,
                        other_data_mat,
                        other_label_mat,
                        self.input_size,
                    )
        else:
            # 最后几个不使用数据增强的迭代
            for _ in range(len(self.file_list)):
                _, data_mat, label_mat = self.rand_read()
                # 迭代 iter_times 次
                for _ in range(self.iter_times):
                    yield get_las_mat(
                        data_mat,
                        label_mat,
                        self.input_size,
                    )

    def rand_read(self):
        """随机读取一个文件的数据和标签"""
        file_idx = torch.randint(0, len(self.file_list), (1,))[0]
        file_name = self.file_list[file_idx]
        file_path = os.path.join(self.data_folder, file_name)
        label_path = os.path.join(
            self.label_folder, file_name.replace(".las", ".label")
        )
        teach_path = os.path.join(
            self.label_folder, file_name.replace(".las", ".teach")
        )
        las_data = read_las_file(file_path)
        # 加载数据
        data_mat = generate_data(las_data, self.voxel_params, self.device)
        label_mat = torch.load(label_path).to(self.device)
        teach_mat = torch.load(teach_path).to(self.device)
        return file_name, data_mat, LabelMat(label_mat, teach_mat)



class TestLoader(IterableDataset):
    """
    训练阶段的多文件训练数据集
    """

    def __init__(
        self,
        root_folder: str,
        voxel_params: VoxelParameters,
        device: str,
    ):
        self.data_folder = os.path.join(root_folder, "data")
        self.label_folder = os.path.join(root_folder, "label")
        self.file_list = [f for f in os.listdir(self.data_folder) if f.endswith(".las")]
        self.voxel_params = voxel_params
        self.device = device

    def __len__(self):
        return len(self.file_list)

    def __iter__(self):
        """顺序读取一个文件的数据和标签"""
        for file_idx in range(len(self.file_list)):
            file_name = self.file_list[file_idx]
            file_path = os.path.join(self.data_folder, file_name)
            label_path = os.path.join(
                self.label_folder, file_name.replace(".las", ".label")
            )
            las_data = read_las_file(file_path)
            # 加载数据
            data_mat = generate_data(las_data, self.voxel_params, self.device)
            label_mat = torch.load(label_path).to(self.device)

            yield (data_mat.input_mat, data_mat.valid_len_mat, label_mat)
