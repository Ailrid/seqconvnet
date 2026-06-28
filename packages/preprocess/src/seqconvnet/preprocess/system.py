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
    TrainingPreprocessMessage,
    MaePreprocessMessage,
)
from .component import PreprocessConfig, PreprocessInfo
from seqconvnet.core import (
    label_map,
    delete_labels,
    preprocess_las_file,
    mae_preprocess_las_file,
)

from tqdm import tqdm
from virid.core import system, ViridApp
from virid.std import execute_block


@system(message_type=MappingMessage)
def mapping(config: PreprocessConfig, app: ViridApp):
    """生成Map"""
    mapping, num_classes, classes_weight = label_map(config.train_las_folder)

    app.spawn(PreprocessInfo(mapping=mapping, num_classes=num_classes))
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
def training_preprocess(
    message: TrainingPreprocessMessage, config: PreprocessConfig, info: PreprocessInfo
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


@system()
def mae_preprocess(
    message: MaePreprocessMessage, config: PreprocessConfig, info: PreprocessInfo
):

    print(f"========== Start mae preprocessing {message.las_folder} ==========")
    data_folder = os.path.join(config.preprocessed_folder, message.las_folder, "data")
    file_list = [f for f in os.listdir(message.las_folder) if f.endswith(".las")]

    with tqdm(file_list, desc="Preprocessing", total=len(file_list)) as pbar:
        for file_name in pbar:
            file_path = os.path.join(message.las_folder, file_name)
            mae_preprocess_las_file(
                file_path,
                data_folder,
                info.mapping,
                config.area_size,
                message.overlap,
                config.voxel_params,
                config.device,
            )


@system()
def start_up(message: StartUpMessage, app: ViridApp):
    """启动预处理"""
    app.spawn(
        PreprocessConfig(
            train_las_folder=message.train_las_folder,
            test_las_folder=message.test_las_folder,
            preprocessed_folder=message.preprocessed_folder,
            area_size=message.area_size,
            voxel_params=message.voxel_params,
            device=message.device,
        )
    )
    # 默认训练参数

    print(
        f"========== Start preprocessing {message.train_las_folder} and {message.test_las_folder} =========="
    )
    print(f"========== Preprocess parameters ==========")
    print(
        f"Min Rows: {message.voxel_params.min_rows}, Min Cols: {message.voxel_params.min_cols}, Max Z: {message.voxel_params.max_z}"
    )
    print(
        f"XY Resolution: {message.voxel_params.xy_resolution}, Z Resolution: {message.voxel_params.z_resolution}"
    )
    print(f"Area Size: {message.area_size} m")
    print(f"Device: {message.device}")

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

        if message.preprocess_target == "mae":
            MaePreprocessMessage.send(message.train_las_folder, True)
            MaePreprocessMessage.send(message.test_las_folder, False)

        elif message.preprocess_target == "training":
            MappingMessage.send()
            TrainingPreprocessMessage.send(message.train_las_folder, True)
            TrainingPreprocessMessage.send(message.test_las_folder, False)

        else:
            raise ValueError('preprocess_target must be "mae" or "training"')


def register_systems(app: ViridApp):
    app.register(start_up)
    app.register(mapping)
    app.register(delete_cls)
    app.register(training_preprocess)
    app.register(mae_preprocess)
