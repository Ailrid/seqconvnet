"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from virid.core import create_virid
from virid.std import StdPlugin
from seqconvnet.train import (
    bind_components,
    DatasetParameters,
    ModelParameters,
    EnvParameters,
    register_systems,
    TrainingLightingMessage,
)
from seqconvnet.core import VoxelParameters

virid = create_virid(max_depth=1000, enable_logging=False).use(StdPlugin, None)

bind_components(virid)
register_systems(virid)

# 点火启动
TrainingLightingMessage.send(
    dataset_params=DatasetParameters(
        train_las_folder="preprocessed/dales_las/train",
        test_las_folder="preprocessed/dales_las/test",
        # classes从1开始，总共8个权重
        num_classes=8,
        classes_weights=[  # 每个类的类别权重
            1.0000,
            1.2866,
            3.2652,
            8.8057,
            7.3190,
            3.9489,
            8.2722,
            1.2322,
        ],
        batch_size=1,  # 建议设置为1
        num_workers=4,  # 数据加载线程
        input_size=128,  # 输入给网络的大小，单位是像素，每块的长和宽就是 input_size * xy_resolution
        area_size=128,  # 点云切块的大小，单位是米
        iter_times=1,  # 每个切块点云上的迭代次数
        # 默认训练参数
        voxel_params=VoxelParameters(
            xy_resolution=0.5,  # 体素化时在xy平面上的分辨率，建议设置为0.5
            z_resolution=0.5,  # 体素化时沿着z轴上的分辨率，建议设置为0.5
            max_z=128,  # 最高的高度是 max_z * z_resolution，超出就会被截断
            min_rows=128,  # 建议设置为1
            min_cols=128,  # 建议设置为1
        ),
    ),
    model_params=ModelParameters(
        checkpoint_folder="checkpoints/2026-06-28-13-55-36",
        model_type="transformer",
        d_model=16,  # transformer 的 d_model 或者 rnn 的 hidden_size
        nhead=2,  # transformer 的 nhead
        num_layers=2,  # transformer 的 num_layers
        dropout=0.1,
    ),
    env_params=EnvParameters(
        lr=1e-3,
        epochs=100,
        warmup_epochs=5,
        weight_decay=1e-2,
        device="cuda:0",
    ),
)

if __name__ == "__main__":
    virid.tick()
