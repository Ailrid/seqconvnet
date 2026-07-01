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
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR, SequentialLR
from virid.core import system, ViridApp, MessageWriter

from seqconvnet.core import (
    SoftDiceAndFocalLoss,
    SwinEncoder,
    CustomConvEncoder,
    RnnClassifier,
    RnnDecoder,
    RnnEncoder,
    RnnChunkShell,
    SegmentationEvaluator,
    TestLoader,
    TrainLoader,
)
from seqconvnet.core import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerClassifier,
    TransformerShell,
    StandardHeightEmbedding,
    HybridHeightEmbedding,
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
from ..util import create_logger, save_light_params, get_warmup_exponential_lambda


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
        "============ Create Logger And Checkpoint Done ============ \n"
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
        TrainLoader(
            dataset_params.train_las_folder,
            dataset_params.iter_times,
            dataset_params.input_size,
            dataset_params.voxel_params,
        ),
        batch_size=dataset_params.batch_size,
        num_workers=dataset_params.num_workers,
    )
    test_loader = DataLoader(
        TestLoader(
            dataset_params.test_las_folder,
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
            num_classes=dataset_params.num_classes,
            num_workers=dataset_params.num_workers,
        )
    )
    MessageWriter.info(
        "============ Create Dataset Done ============ \n"
        f"Train Las Folder: {dataset_params.train_las_folder}\n"
        f"Test Las Folder: {dataset_params.test_las_folder}\n"
        f"Num Classes: {dataset_params.num_classes}\n"
        f"Iter Times: {dataset_params.iter_times}\n"
        f"Input Size: {dataset_params.input_size}\n"
        f"Area Size: {dataset_params.area_size}\n"
        f"Voxel Params: {dataset_params.voxel_params}\n"
        f"Classes Weights: {dataset_params.classes_weights}\n"
    )


@system(message_type=CreateTransformerMessage)
def create_transformer_model(
    app: ViridApp,
    light_params: LightParameters,
) -> None:
    model_params = light_params.model_params
    env_params = light_params.env_params
    dataset_params = light_params.dataset_params
    # embedding = HybridHeightEmbedding(
    #     dataset_params.voxel_params.max_z,
    #     model_params.d_model,
    # )
    embedding = StandardHeightEmbedding(
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
        dataset_params.num_classes,
        model_params.d_model,
    )

    model = TransformerShell(
        embedding, seq_encoder, conv_encoder, seq_decoder, classifier
    ).to(env_params.device)

    loss = SoftDiceAndFocalLoss(
        dataset_params.num_classes,
        dataset_params.classes_weights,
    ).to(env_params.device)

    app.spawn(
        ModelConfig(
            model=model,
            loss=loss,
        )
    )
    MessageWriter.info(
        "============ Create Transformer Model Done ============ \n"
        f"Max Z: { dataset_params.voxel_params.max_z}\n"
        f"Dim Model: {model_params.d_model}\n"
        f"Num Head: {model_params.nhead}\n"
        f"NUm Layers: {model_params.num_layers}\n"
        f"Dropout: {model_params.dropout}\n"
        f"Classes Weights: {dataset_params.classes_weights}\n"
    )


@system(message_type=CreateRnnMessage)
def create_rnn_model(
    app: ViridApp,
    light_params: LightParameters,
) -> None:
    model_params = light_params.model_params
    env_params = light_params.env_params
    dataset_params = light_params.dataset_params
    # 组装网络
    encoder_embedding = StandardHeightEmbedding(
        dataset_params.voxel_params.max_z,
        model_params.d_model,
    )
    # 这里 + 2, max_z + 2 给 BOS 一个位置, 0 给 PAD 一个位置
    decoder_embedding = torch.nn.Embedding(
        dataset_params.voxel_params.max_z + 2, model_params.d_model
    )

    seq_encoder = RnnEncoder(
        model_params.d_model,
        model_params.d_model,
        model_params.num_layers,
        model_params.dropout,
    )

    # conv_encoder = SwinEncoder(
    #     2 * model_params.d_model, 2 * model_params.d_model, dataset_params.input_size
    # )
    conv_encoder = CustomConvEncoder(2 * model_params.d_model, 2 * model_params.d_model)

    seq_decoder = RnnDecoder(
        model_params.d_model,
        model_params.d_model,
        model_params.num_layers,
        model_params.dropout,
    )

    classifier = RnnClassifier(
        dataset_params.num_classes,
        model_params.d_model,
    )

    model = RnnChunkShell(
        encoder_embedding,
        decoder_embedding,
        seq_encoder,
        conv_encoder,
        seq_decoder,
        classifier,
    ).to(env_params.device)

    loss = SoftDiceAndFocalLoss(
        dataset_params.num_classes,
        dataset_params.classes_weights,
    ).to(env_params.device)

    app.spawn(
        ModelConfig(
            model=model,
            loss=loss,
        )
    )
    MessageWriter.info(
        "============ Create Rnn Model Done ============ \n"
        f"Max Z: { dataset_params.voxel_params.max_z}\n"
        f"Dim Model: {model_params.d_model}\n"
        f"Num Layers: {model_params.num_layers}\n"
        f"Dropout: {model_params.dropout}\n"
        f"Classes Weights: {dataset_params.classes_weights}\n"
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
    # 先初始化余弦
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(env_params.epochs - env_params.warmup_epochs),
        eta_min=5e-7,
    )

    # 后初始化 LinearLR
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=env_params.warmup_epochs,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[env_params.warmup_epochs],
    )
    # lr_lambda = get_warmup_exponential_lambda(
    #     warmup_epochs=env_params.warmup_epochs,
    #     start_factor=0.1,
    #     eta_min=1e-7,
    #     base_lr=env_params.lr,
    #     gamma=0.90,
    # )
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    evaluator = SegmentationEvaluator(light_params.dataset_params.num_classes, True)
    app.spawn(
        EnvConfig(
            lr=env_params.lr,
            epochs=env_params.epochs,
            warmup_epochs=env_params.warmup_epochs,
            evaluator=evaluator,
            scheduler=scheduler,
            optimizer=optimizer,
            device=env_params.device,
        )
    )

    MessageWriter.info(
        "============ Create Train Env Done ============ \n"
        f"Lr: {env_params.lr}\n"
        f"Weight Decay: {env_params.weight_decay}\n"
        f"Epochs: {env_params.epochs}\n"
        f"Warmup Epochs: {env_params.warmup_epochs}\n"
        f"Device: {env_params.device}, Device Name: {torch.cuda.get_device_name(env_params.device)}\n"
    )


def register_initialization_systems(app: ViridApp):
    app.register(create_dataset)
    app.register(create_transformer_model)
    app.register(create_rnn_model)
    app.register(create_env)
    app.register(create_logger_and_checkpoint)
