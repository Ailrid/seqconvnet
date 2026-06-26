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
)
from ..messages.training import (
    StartTrainingMessage,
)
from ..components import ModelConfig, EnvConfig, DatasetConfig
from virid.core import system, ViridApp
from virid.std import execute_block

from torch.utils.data import DataLoader


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
    print("============ Start Up Training ============")
    print(f"Model Params: {message.model_params}")
    print(f"Training Params: {message.env_params}")
    print(f"Dataset Params: {message.dataset_params}")

    def callback(success: bool):
        if success:
            print("Training initialized successfully")
        else:
            print("Training initialized failed")

    with execute_block(group_id="startup", callback=callback):

        if message.model_params.checkpoint_folder != "":
            CreateTransformerMessage.send(
                message.dataset_params,
                message.model_params,
                message.env_params,
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


@system()
def create_dataset(
    message: CreateDatasetMessage, dataset_config: DatasetConfig
) -> None:
    dataset_params = message.dataset_params

    train_loader = DataLoader(
        TrainLoader(
            dataset_params.train_las_folder,
            dataset_params.iter_times,
            dataset_params.input_size,
            dataset_params.voxel_params,
            message.env_params.device,
        ),
        batch_size=dataset_params.batch_size,
        num_workers=dataset_params.num_workers,
    )
    test_loader = DataLoader(
        TestLoader(
            dataset_params.test_las_folder,
            dataset_params.voxel_params,
            message.env_params.device,
        ),
        batch_size=dataset_params.batch_size,
        num_workers=dataset_params.num_workers,
    )
    dataset_config.train_loader = train_loader
    dataset_config.test_loader = test_loader
    dataset_config.num_classes = dataset_params.num_classes
    dataset_config.voxel_params = dataset_params.voxel_params
    dataset_config.input_size = dataset_params.input_size

    print("============ Create Dataset Done ============ ")
    print(f"Train Las Folder: {dataset_params.train_las_folder}")
    print(f"Test Las Folder: {dataset_params.test_las_folder}")
    print(f"Num Classes: {dataset_params.num_classes}")
    print(f"Iter Times: {dataset_params.iter_times}")
    print(f"Input Size: {dataset_params.input_size}")
    print(f"Area Size: {dataset_params.area_size}")
    print(f"Voxel Params: {dataset_params.voxel_params}")
    print(f"Classes Weights: {dataset_params.classes_weights}")


@system()
def create_transformer_model(
    message: CreateTransformerMessage, config: ModelConfig
) -> None:
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
    )
    conv_encoder = SwinEncoder(
        model_params.d_model, model_params.d_model, dataset_params.input_size
    )
    seq_decoder = TransformerDecoder(
        dataset_params.num_classes,
        model_params.d_model,
        model_params.nhead,
        model_params.num_layers,
        model_params.dropout,
    )
    classifier = TransformerClassifier(
        model_params.d_model,
        dataset_params.num_classes,
    )

    model = TransformerShell(seq_encoder, conv_encoder, seq_decoder, classifier).to(
        env_params.device
    )
    # 尝试加载检查点
    # model.load_checkpoint(model_params.checkpoint_folder)
    config.model = model
    config.loss = SoftDiceAndFocalLoss(
        dataset_params.num_classes,
        dataset_params.classes_weights,
        device=env_params.device,
    )
    print("============ Create Transformer Model Done ============ ")
    print(f"Max Z: { dataset_params.voxel_params.max_z}")
    print(f"Dim Model: {model_params.d_model}")
    print(f"Num Head: {model_params.nhead}")
    print(f"NUm Layers: {model_params.num_layers}")
    print(f"Dropout: {model_params.dropout}")
    print(f"Classes Weights: {dataset_params.classes_weights}")


@system()
def create_model(message: CreateRnnMessage, config: ModelConfig) -> None:
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
    )
    conv_encoder = CustomConvEncoder(
        model_params.num_layers * model_params.d_model,
        model_params.num_layers * model_params.d_model,
    )
    seq_decoder = RnnDecoder(
        dataset_params.num_classes,
        2 * model_params.d_model,
        model_params.d_model,
        model_params.num_layers,
        model_params.dropout,
    )
    classifier = RnnClassifier(
        dataset_params.num_classes,
        model_params.d_model,
    )

    model = RnnShell(seq_encoder, conv_encoder, seq_decoder, classifier).to(
        env_params.device
    )
    # 尝试加载检查点
    config.model = model
    config.loss = SoftDiceAndFocalLoss(
        dataset_params.num_classes,
        dataset_params.classes_weights,
        device=env_params.device,
    )
    print("============ Create Rnn Model Done ============ ")
    print(f"Max Z: { dataset_params.voxel_params.max_z}")
    print(f"Dim Model: {model_params.d_model}")
    print(f"Num Head: {model_params.nhead}")
    print(f"NUm Layers: {model_params.num_layers}")
    print(f"Dropout: {model_params.dropout}")
    print(f"Classes Weights: {dataset_params.classes_weights}")


@system()
def create_env(
    message: CreateEvnMessage, env_config: EnvConfig, model_config: ModelConfig
) -> None:
    env_params = message.env_params
    env_config.epochs = env_params.epochs
    env_config.warmup_epochs = env_params.warmup_epochs
    env_config.lr = env_params.lr
    env_config.device = env_params.device
    env_config.evaluator = SegmentationEvaluator(
        message.dataset_params.num_classes, True
    )
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
    env_config.scheduler = scheduler
    env_config.optimizer = optimizer
    print("============ Create Train Env Done ============ ")
    print(f"Lr: {env_params.lr}")
    print(f"Weight Decay: {env_params.weight_decay}")
    print(f"Epochs: {env_params.epochs}")
    print(f"Warmup Epochs: {env_params.warmup_epochs}")
    print(
        f"Device: {env_params.device}, Device Name: {torch.cuda.get_device_name(env_params.device)}"
    )


def register_initialization_systems(app: ViridApp):
    app.register(start_up)
    app.register(create_dataset)
    app.register(create_model)
    app.register(create_env)
