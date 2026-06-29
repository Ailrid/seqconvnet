"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import time
import os
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from virid.core import system, ViridApp, MessageWriter

from seqconvnet.core import (
    MaeVoxelLoss,
    SwinEncoder,
    CustomConvEncoder,
    RnnClassifier,
    RnnDecoder,
    RnnEncoder,
    RnnShell,
    MaeTestLoader,
    MaeTrainLoader,
)
from seqconvnet.core import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerClassifier,
    MaeTransformerShell,
    MaskedHeightEmbedding,
)

from ..messages.initialization import (
    CreateEvnMessage,
    CreateTransformerMessage,
    CreateRnnMessage,
    CreateDatasetMessage,
    CreateLoggerAndCheckpointMessage,
)

from ..components import (
    ModelConfig,
    EnvConfig,
    DatasetConfig,
    TrainingLogger,
    TrainingState,
    LightParameters,
)
from ..util import create_logger, save_light_params


@system(message_type=CreateLoggerAndCheckpointMessage)
def create_logger_and_checkpoint(
    logger: TrainingLogger,
    training_state: TrainingState,
    light_params: LightParameters,
):
    # 创建开始时间文件夹
    logger_folder = os.path.join(
        "./logs", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    )
    os.makedirs(logger_folder, exist_ok=True)
    # 创建日志文件
    log_file_path = os.path.join(logger_folder, "training.txt")
    logger.writer = create_logger(log_file_path)
    # 创建checkpoint文件夹
    checkpoint_folder = os.path.join(
        "./checkpoints",
        # 精确到秒
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
    )
    os.makedirs(checkpoint_folder, exist_ok=True)
    training_state.log_folder = logger_folder
    training_state.checkpoint_folder = checkpoint_folder
    # 保存一份训练设置到checkpoint文件夹中
    save_light_params(checkpoint_folder, light_params)
    MessageWriter.info(
        "============ Create Mae Logger And Checkpoint Done ============ \n"
        f"Logger Folder: {logger_folder}\n"
        f"Checkpoint Folder: {checkpoint_folder}\n"
    )


@system(message_type=CreateDatasetMessage)
def create_dataset(
    app: ViridApp,
    light_params: LightParameters,
) -> None:
    dataset_params = light_params.dataset_params

    train_loader = DataLoader(
        MaeTrainLoader(
            dataset_params.train_las_folder,
            dataset_params.mask_ratio,
            dataset_params.iter_times,
            dataset_params.input_size,
            dataset_params.voxel_params,
        ),
        batch_size=dataset_params.batch_size,
        num_workers=dataset_params.num_workers,
    )
    test_loader = DataLoader(
        MaeTestLoader(
            dataset_params.test_las_folder,
            dataset_params.mask_ratio,
            dataset_params.voxel_params,
        ),
        batch_size=dataset_params.batch_size,
    )
    app.spawn(
        DatasetConfig(
            batch_size=dataset_params.batch_size,
            input_size=dataset_params.input_size,
            train_loader=train_loader,
            test_loader=test_loader,
            voxel_params=dataset_params.voxel_params,
            num_workers=dataset_params.num_workers,
            mask_ratio=dataset_params.mask_ratio,
        )
    )
    MessageWriter.info(
        "============ Create Mae Dataset Done ============ \n"
        f"Train Las Folder: {dataset_params.train_las_folder}\n"
        f"Test Las Folder: {dataset_params.test_las_folder}\n"
        f"Iter Times: {dataset_params.iter_times}\n"
        f"Input Size: {dataset_params.input_size}\n"
        f"Area Size: {dataset_params.area_size}\n"
        f"Voxel Params: {dataset_params.voxel_params}\n"
    )


@system(message_type=CreateTransformerMessage)
def create_transformer_model(
    app: ViridApp,
    light_params: LightParameters,
) -> None:
    model_params = light_params.model_params
    env_params = light_params.env_params
    dataset_params = light_params.dataset_params
    embedding = MaskedHeightEmbedding(
        dataset_params.voxel_params.max_z,
        model_params.d_model,
    )
    # 组装网络
    seq_encoder = TransformerEncoder(
        model_params.d_model,
        model_params.nhead,
        model_params.num_layers,
        model_params.dropout,
    )

    conv_encoder = SwinEncoder(
        model_params.d_model, model_params.d_model, dataset_params.input_size
    )

    # conv_encoder = CustomConvEncoder(
    #     model_params.d_model,
    #     model_params.d_model,
    # )

    seq_decoder = TransformerDecoder(
        model_params.d_model,
        model_params.nhead,
        model_params.num_layers,
        model_params.dropout,
    )

    classifier = TransformerClassifier(
        dataset_params.voxel_params.max_z,
        model_params.d_model,
    )

    model = MaeTransformerShell(
        embedding, seq_encoder, conv_encoder, seq_decoder, classifier
    ).to(env_params.device)

    loss = MaeVoxelLoss(
        dataset_params.voxel_params.max_z,
    )
    app.spawn(
        ModelConfig(
            model=model,
            loss=loss,
        )
    )
    MessageWriter.info(
        "============ Create Mae Transformer Model Done ============ \n"
        f"Max Z: { dataset_params.voxel_params.max_z}\n"
        f"Dim Model: {model_params.d_model}\n"
        f"Num Head: {model_params.nhead}\n"
        f"NUm Layers: {model_params.num_layers}\n"
        f"Dropout: {model_params.dropout}\n"
    )


@system(message_type=CreateEvnMessage)
def create_env(
    app: ViridApp,
    model_config: ModelConfig,
    light_params: LightParameters,
) -> None:
    env_params = light_params.env_params

    optimizer = optim.AdamW(
        model_config.model.parameters(),
        lr=env_params.lr,
        weight_decay=env_params.weight_decay,
    )
    # Warmup 阶段
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=env_params.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(env_params.epochs - env_params.warmup_epochs),
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[env_params.warmup_epochs],
    )

    app.spawn(
        EnvConfig(
            lr=env_params.lr,
            epochs=env_params.epochs,
            warmup_epochs=env_params.warmup_epochs,
            scheduler=scheduler,
            optimizer=optimizer,
            device=env_params.device,
        )
    )

    MessageWriter.info(
        "============ Create Mae Train Env Done ============ \n"
        f"Lr: {env_params.lr}\n"
        f"Weight Decay: {env_params.weight_decay}\n"
        f"Epochs: {env_params.epochs}\n"
        f"Warmup Epochs: {env_params.warmup_epochs}\n"
        f"Device: {env_params.device}, Device Name: {torch.cuda.get_device_name(env_params.device)}\n"
    )


def register_initialization_systems(app: ViridApp):
    app.register(create_dataset)
    app.register(create_transformer_model)
    app.register(create_env)
    app.register(create_logger_and_checkpoint)
