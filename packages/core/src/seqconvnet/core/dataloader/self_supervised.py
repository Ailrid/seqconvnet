# """
# Copyright (c) 2026-present Ailrid.
# Licensed under the Apache License, Version 2.0.
# Project: seqconvnet
# """

# import os
# import torch
# from ..utils.train import enhance, get_las_mat
# from ..structs import VoxelParameters
# from torch.utils.data import IterableDataset


# class MaeTrainLoader(IterableDataset):
#     """
#     训练阶段的多文件训练数据集
#     """

#     def __init__(
#         self,
#         root_folder: str,
#         mask_ratio: float,
#         iter_times: int,
#         input_size: int,
#         voxel_params: VoxelParameters,
#     ):
#         self.eos = voxel_params.max_z + 2
#         self.mask_id = voxel_params.max_z + 3

#         self.mask_ratio = mask_ratio
#         self.iter_times = iter_times
#         self.data_folder = os.path.join(root_folder, "data")
#         self.file_list = [
#             f for f in os.listdir(self.data_folder) if f.endswith(".input")
#         ]
#         self.input_size = input_size
#         self.voxel_params = voxel_params
#         self.enhance_key = True

#     def toggle_enhance(self):
#         self.enhance_key = not self.enhance_key

#     def __len__(self):
#         return len(self.file_list) * self.iter_times

#     def __iter__(self):
#         if self.enhance_key:
#             for _ in range(len(self.file_list)):
#                 data_mat, valid_len_mat = self.rand_read()
#                 other_data_mat, other_valid_len_mat = self.rand_read()
#                 # 迭代 iter_times 次
#                 for _ in range(self.iter_times):
#                     enhance_input_mat, enhance_valid_len_mat, _, _ = enhance(
#                         data_mat,
#                         valid_len_mat,
#                         valid_len_mat,
#                         valid_len_mat,
#                         other_data_mat,
#                         other_valid_len_mat,
#                         other_valid_len_mat,
#                         other_valid_len_mat,
#                         self.input_size,
#                     )
#                     yield enhance_input_mat, enhance_valid_len_mat
#         else:
#             # 最后几个不使用数据增强的迭代
#             for _ in range(len(self.file_list)):
#                 data_mat, valid_len_mat = self.rand_read()
#                 # 迭代 iter_times 次
#                 for _ in range(self.iter_times):
#                     area_input_mat, area_valid_len_mat, _, _ = get_las_mat(
#                         data_mat,
#                         valid_len_mat,
#                         valid_len_mat,
#                         valid_len_mat,
#                         self.input_size,
#                     )
#                     mask_input_mat = self.mask(area_input_mat)
#                     yield mask_input_mat, area_valid_len_mat, area_input_mat

#     def rand_read(self):
#         """随机读取一个文件的数据和标签"""
#         file_idx = torch.randint(0, len(self.file_list), (1,))[0]
#         file_name = self.file_list[file_idx]
#         data_path = os.path.join(self.data_folder, file_name)
#         valid_len_path = os.path.join(
#             self.data_folder, file_name.replace(".input", ".valid_len")
#         )

#         # 加载数据
#         data_mat = torch.load(data_path)
#         valid_len_mat = torch.load(valid_len_path)
#         return data_mat, valid_len_mat

#     def mask(self, input_mat):
#         # 找出哪些位置既不是空气(>0)，也不是结束符(<eos)
#         without_eos_mask = (input_mat < self.eos) & (input_mat > 0)  # [B * H * W, S]
#         # 生成与输入形状一致的随机矩阵
#         rand_matrix = torch.rand(input_mat.shape, device=input_mat.device)

#         # 只有在有效体素且随机数小于掩码率的地方，才触发真正的 MAE 遮罩
#         mae_mask = without_eos_mask & (rand_matrix < self.mask_ratio)  # [B * H * W, S]

#         masked_seq_input = input_mat.clone()
#         masked_seq_input[mae_mask] = self.mask_id
#         return masked_seq_input


# class MaeTestLoader(IterableDataset):
#     """
#     训练阶段的多文件训练数据集
#     """

#     def __init__(
#         self,
#         root_folder: str,
#         mask_ratio: float,
#         voxel_params: VoxelParameters,
#     ):
#         self.eos = voxel_params.max_z + 2
#         self.mask_id = voxel_params.max_z + 3
#         self.mask_ratio = mask_ratio
#         self.data_folder = os.path.join(root_folder, "data")
#         self.label_folder = os.path.join(root_folder, "label")
#         self.file_list = [
#             f for f in os.listdir(self.data_folder) if f.endswith(".input")
#         ]
#         self.voxel_params = voxel_params

#     def __len__(self):
#         return len(self.file_list)

#     def __iter__(self):
#         """顺序读取一个文件的数据和标签"""
#         for file_idx in range(len(self.file_list)):
#             file_name = self.file_list[file_idx]
#             data_path = os.path.join(self.data_folder, file_name)

#             # 加载数据
#             input_mat = torch.load(data_path)
#             valid_len_mat = (input_mat != 0).to(torch.float32)

#             yield (self.mask(input_mat), valid_len_mat, input_mat)

#     def mask(self, input_mat):
#         # 找出哪些位置既不是空气(>0)，也不是结束符(<eos)
#         without_eos_mask = (input_mat < self.eos) & (input_mat > 0)  # [B * H * W, S]
#         # 生成与输入形状一致的随机矩阵
#         rand_matrix = torch.rand(input_mat.shape, device=input_mat.device)

#         # 只有在有效体素且随机数小于掩码率的地方，才触发真正的 MAE 遮罩
#         mae_mask = without_eos_mask & (rand_matrix < self.mask_ratio)  # [B * H * W, S]

#         masked_seq_input = input_mat.clone()
#         masked_seq_input[mae_mask] = self.mask_id
#         return masked_seq_input
