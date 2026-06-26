"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import os
from .message import (
    # LinkTestDataMessage,
    StartUpMessage,
    MappingMessage,
    DeleteLabelsMessage,
    PreprocessMessage,
)
from .component import PreprocessConfig, PreprocessInfo
from seqconvnet.core import (
    label_map,
    delete_labels,
    preprocess_las_file,
)

from tqdm import tqdm
from virid.core import system, ViridApp
from virid.std import execute_block


@system(message_type=MappingMessage)
def mapping(config: PreprocessConfig, info: PreprocessInfo):
    """生成Map"""
    mapping, num_classes, classes_weight = label_map(config.train_las_folder)
    info.mapping = mapping
    info.num_classes = num_classes
    # 打印标签映射关系
    print(
        "========== Remap label values, with the following mapping relationship =========="
    )
    print("Label Mapping:")
    for old_label, new_label in mapping.items():
        print(f"Old label: {old_label} -> New label: {new_label}")
    print(f"Count Classes: {num_classes}")
    print(f"Class Weights: {classes_weight}")


@system()
def delete_cls(message: DeleteLabelsMessage):
    """删除指定的类别"""
    print(f"========== Start deleting labels in {message.las_folder} ==========")
    file_list = [f for f in os.listdir(message.las_folder) if f.endswith(".las")]

    with tqdm(file_list, desc="Deleting label", total=len(file_list)) as pbar:
        for file_name in pbar:
            las_path = os.path.join(message.las_folder, file_name)
            delete_labels(las_path, message.delete_labels)

    for label in message.delete_labels:
        print(f"Delete label: {label}")


@system()
def preprocess(
    message: PreprocessMessage, config: PreprocessConfig, info: PreprocessInfo
):

    print(f"========== Start preprocessing {message.las_folder} ==========")
    data_folder = os.path.join(config.preprocessed_folder, message.las_folder, "data")
    label_folder = os.path.join(config.preprocessed_folder, message.las_folder, "label")
    file_list = [f for f in os.listdir(message.las_folder) if f.endswith(".las")]

    with tqdm(file_list, desc="Preprocessing", total=len(file_list)) as pbar:
        for file_name in pbar:
            file_path = os.path.join(message.las_folder, file_name)
            preprocess_las_file(
                file_path,
                data_folder,
                label_folder,
                info.num_classes,
                info.mapping,
                config.area_size,
                message.overlap,
                config.voxel_params,
                config.device,
            )


# @system()
# def link_test_data(message: LinkTestDataMessage, config: PreprocessConfig):
#     print(f"========== Start preprocessing {message.las_folder} ==========")
#     data_folder = os.path.join(config.preprocessed_folder, message.las_folder, "data")
#     os.makedirs(data_folder, exist_ok=True)
#     file_list = [f for f in os.listdir(message.las_folder) if f.endswith(".las")]
#     for file_name in file_list:
#         file_path = os.path.join(message.las_folder, file_name)
#         link_path = os.path.join(data_folder, file_name)
#         if os.path.exists(link_path):
#             os.remove(link_path)
#         os.symlink(file_path, link_path)


@system()
def start_up(message: StartUpMessage, config: PreprocessConfig):
    """启动预处理"""
    config.train_las_folder = message.train_las_folder
    config.test_las_folder = message.test_las_folder
    config.preprocessed_folder = message.preprocessed_folder
    # 默认训练参数
    config.voxel_params = message.voxel_params
    config.area_size = message.area_size
    config.device = message.device
    print(
        f"========== Start preprocessing {message.train_las_folder} and {message.test_las_folder} =========="
    )
    print(f"========== Preprocess parameters ==========")
    print(
        f"Min Rows: {config.voxel_params.min_rows}, Min Cols: {config.voxel_params.min_cols}, Max Z: {config.voxel_params.max_z}"
    )
    print(
        f"XY Resolution: {config.voxel_params.xy_resolution}, Z Resolution: {config.voxel_params.z_resolution}"
    )
    print(f"Area Size: {config.area_size} m")
    print(f"Device: {config.device}")

    def callback(success: bool):
        if success:
            print("Preprocess finished")
        else:
            print("Preprocess failed")

    with execute_block(group_id="start_up", callback=callback):
        # 如果要删除类别，那么就先删除该类别
        if message.delete_labels is not None:
            DeleteLabelsMessage.send(message.train_las_folder, message.delete_labels)
            DeleteLabelsMessage.send(message.test_las_folder, message.delete_labels)
        MappingMessage.send()
        PreprocessMessage.send(message.train_las_folder, True)
        # LinkTestDataMessage.send(message.test_las_folder)
        PreprocessMessage.send(message.test_las_folder, False)


def register_systems(app: ViridApp):
    app.register(start_up)
    app.register(mapping)
    app.register(delete_cls)
    app.register(preprocess)
