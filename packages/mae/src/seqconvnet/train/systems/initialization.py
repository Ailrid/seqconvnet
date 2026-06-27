"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from seqconvnet.core import (
    SoftDiceAndFocalLoss,
    SwinEncoder,
    RnnClassifier,
    RnnDecoder,
    RnnEncoder,
    RnnShell,
    CustomConvEncoder,
)
from seqconvnet.core import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerClassifier,
    TransformerShell,
)
from seqconvnet.core import SegmentationEvaluator, TestLoader, TrainLoader
from torch import optim
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from ..messages.initialization import (
    StartUpMessage,
    CreateEvnMessage,
    CreateTransformerMessage,
    CreateRnnMessage,
    CreateDatasetMessage,
    CreateLoggerAndCheckpointMessage,
)
from ..messages.training import (
    StartTrainingMessage,
)
from ..components import (
    ModelConfig,
    EnvConfig,
    DatasetConfig,
    TrainingLogger,
    TrainingState,
)
from virid.core import system, ViridApp, MessageWriter
from virid.std import execute_block
from torch.utils.data import DataLoader
from ..util import create_logger
import time
import os
import json
from dataclasses import asdict


class Color:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ORANGE = "\033[33m"
    GREY = "\033[90m"
    BOLD = "\033[1m"
    END = "\033[0m"  # 用来结束颜色，否则后面的文本都会变色


@system()
def start_up(message: StartUpMessage):
    """启动训练流程"""

    def callback(success: bool):
        if success:
            MessageWriter.info("Training initialized successfully")
        else:
            MessageWriter.error(RuntimeError("Training initialized failed"))

    with execute_block(group_id="startup", callback=callback):

        CreateLoggerAndCheckpointMessage.send()
        # 在日志初始化之后才能开始打印
        MessageWriter.info(
            "============ Start Up Training ============ \n"
            f"Model Params: {json.dumps(asdict(message.model_params),indent=4, ensure_ascii=False)}\n"
            f"Training Params: {json.dumps(asdict(message.env_params),indent=4, ensure_ascii=False)}\n"
            f"Dataset Params: {json.dumps(asdict(message.dataset_params),indent=4, ensure_ascii=False)}\n"
        )
        CreateDatasetMessage.send(
            message.dataset_params,
            message.model_params,
            message.env_params,
        )

        if message.model_params.model_type == "transformer":
            CreateTransformerMessage.send(
                message.dataset_params,
                message.model_params,
                message.env_params,
            )
        elif message.model_params.model_type == "rnn":
            CreateRnnMessage.send(
                message.dataset_params,
                message.model_params,
                message.env_params,
            )
        else:
            raise ValueError("Invalid model type")

        CreateEvnMessage.send(
            message.dataset_params,
            message.model_params,
            message.env_params,
        )
        StartTrainingMessage.send()


@system(message_type=CreateLoggerAndCheckpointMessage)
def create_logger_and_checkpoint(
    logger: TrainingLogger,
    training_state: TrainingState,
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

    MessageWriter.info(
        "============ Create Logger And Checkpoint Done ============ \n"
        f"Logger Folder: {logger_folder}\n"
        f"Checkpoint Folder: {checkpoint_folder}\n"
    )


@system()
def create_dataset(message: CreateDatasetMessage, app: ViridApp) -> None:
    dataset_params = message.dataset_params

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


@system()
def create_transformer_model(message: CreateTransformerMessage, app: ViridApp) -> None:
    model_params = message.model_params
    env_params = message.env_params
    dataset_params = message.dataset_params
    # 组装网络
    seq_encoder = TransformerEncoder(
        dataset_params.voxel_params.max_z,
        model_params.d_model,
        model_params.nhead,
        model_params.num_layers,
        model_params.dropout,
    ).to(env_params.device)
    # conv_encoder = SwinEncoder(
    #     model_params.d_model, model_params.d_model, dataset_params.input_size
    # ).to(env_params.device)
    conv_encoder = CustomConvEncoder(
        model_params.d_model,
        model_params.d_model,
    ).to(env_params.device)
    seq_decoder = TransformerDecoder(
        dataset_params.num_classes,
        model_params.d_model,
        model_params.nhead,
        model_params.num_layers,
        model_params.dropout,
    ).to(env_params.device)
    classifier = TransformerClassifier(
        model_params.d_model,
        dataset_params.num_classes,
    ).to(env_params.device)

    model = TransformerShell(seq_encoder, conv_encoder, seq_decoder, classifier).to(
        env_params.device
    )
    loss = SoftDiceAndFocalLoss(
        dataset_params.num_classes,
        dataset_params.classes_weights,
        device=env_params.device,
    )
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


@system()
def create_rnn_model(message: CreateRnnMessage, app: ViridApp) -> None:
    model_params = message.model_params
    env_params = message.env_params
    dataset_params = message.dataset_params
    # 组装网络
    seq_encoder = RnnEncoder(
        dataset_params.voxel_params.max_z,
        model_params.d_model,
        model_params.d_model,
        model_params.num_layers,
        model_params.dropout,
    ).to(env_params.device)
    conv_encoder = CustomConvEncoder(
        model_params.num_layers * model_params.d_model,
        model_params.num_layers * model_params.d_model,
    ).to(env_params.device)
    seq_decoder = RnnDecoder(
        dataset_params.num_classes,
        2 * model_params.d_model,
        model_params.d_model,
        model_params.num_layers,
        model_params.dropout,
    ).to(env_params.device)
    classifier = RnnClassifier(
        dataset_params.num_classes,
        model_params.d_model,
    ).to(env_params.device)

    model = RnnShell(seq_encoder, conv_encoder, seq_decoder, classifier).to(
        env_params.device
    )
    loss = SoftDiceAndFocalLoss(
        dataset_params.num_classes,
        dataset_params.classes_weights,
        device=env_params.device,
    )
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


@system()
def create_env(
    message: CreateEvnMessage, app: ViridApp, model_config: ModelConfig
) -> None:
    env_params = message.env_params

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

    evaluator = SegmentationEvaluator(message.dataset_params.num_classes, True)
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
    app.register(start_up)
    app.register(create_dataset)
    app.register(create_transformer_model)
    app.register(create_rnn_model)
    app.register(create_env)
    app.register(create_logger_and_checkpoint)
