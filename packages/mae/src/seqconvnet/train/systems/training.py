"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import json
import torch
from tqdm import tqdm
import numpy as np
from typing import cast
from dataclasses import asdict
from virid.core import system, ViridApp, MessageWriter
from virid.std import execute_block
from seqconvnet.core import TrainLoader, refer_mat
from seqconvnet.core.utils.pred import yield_input_mat

from ..messages.training import (
    TrainingLightingMessage,
    StartTrainingMessage,
    SaveCheckPointMessage,
    OneEpochMessage,
    EvalMessage,
    LoadMaeCheckPointMessage,
    LoadCheckPointMessage,
)
from ..components import (
    ModelConfig,
    EnvConfig,
    DatasetConfig,
    TrainingState,
    LightParameters,
)
from ..messages.initialization import (
    CreateEvnMessage,
    CreateTransformerMessage,
    CreateRnnMessage,
    CreateDatasetMessage,
    CreateLoggerAndCheckpointMessage,
)
from ..util import confirm_light_params


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
def training_lighting(message: TrainingLightingMessage, app: ViridApp):
    """启动训练流程"""
    # 动态插入 LightParameters 组件
    app.spawn(
        LightParameters(
            message.dataset_params,
            message.model_params,
            message.env_params,
        )
    )

    def callback(success: bool):
        if success:
            MessageWriter.info("Training initialized successfully")
        else:
            MessageWriter.error(RuntimeError("Training initialized failed"))

    with execute_block(group_id="startup", callback=callback):

        CreateLoggerAndCheckpointMessage.send()

        # 在日志初始化之后才能开始打印
        MessageWriter.info(
            "============ Start Up Mae Training ============ \n"
            f"Model Params: {json.dumps(asdict(message.model_params),indent=4, ensure_ascii=False)}\n"
            f"Training Params: {json.dumps(asdict(message.env_params),indent=4, ensure_ascii=False)}\n"
            f"Dataset Params: {json.dumps(asdict(message.dataset_params),indent=4, ensure_ascii=False)}\n"
        )

        CreateDatasetMessage.send()
        # 创建不同模型
        if message.model_params.model_type == "transformer":
            CreateTransformerMessage.send()
        elif message.model_params.model_type == "rnn":
            CreateRnnMessage.send()
        else:
            raise ValueError(
                "Invalid model type, only rnn and transformer are supported"
            )

        CreateEvnMessage.send()
        # 加载检查点
        if (
            message.model_params.checkpoint_folder is not None
            and message.model_params.mae_checkpoint_folder is not None
        ):
            raise ValueError(
                "Cannot use both checkpoint_folder and mae_checkpoint_folder"
            )

        if message.model_params.checkpoint_folder is not None:
            LoadCheckPointMessage.send()

        if message.model_params.mae_checkpoint_folder is not None:
            LoadMaeCheckPointMessage.send()

        StartTrainingMessage.send()


@system(message_type=SaveCheckPointMessage)
def save_checkpoint(
    training_state: TrainingState,
    model_config: ModelConfig,
) -> None:
    checkpoint_folder = training_state.checkpoint_folder
    current_metrics = training_state.current_metrics
    best_metrics = training_state.best_metrics
    # 只保存最好的一轮
    if current_metrics > best_metrics:
        training_state.best_metrics = current_metrics
        model_config.model.save_mae_checkpoint(
            checkpoint_folder, training_state.best_metrics
        )


@system(message_type=LoadCheckPointMessage)
def load_checkpoint(
    light_params: LightParameters,
    model_config: ModelConfig,
) -> None:
    checkpoint_folder = light_params.model_params.checkpoint_folder
    if checkpoint_folder is None:
        return
    check_result = confirm_light_params(checkpoint_folder, light_params)
    model_config.model.load_checkpoint(checkpoint_folder)

    MessageWriter.info(
        "============ Load Mae CheckPoint Successfully ============ \n"
        f"From Checkpoint Folder: {checkpoint_folder}\n"
        f"Confirmed Light Parameters:\n{check_result}\n"
    )


@system(message_type=LoadMaeCheckPointMessage)
def load_mae_checkpoint(
    light_params: LightParameters,
) -> None:
    raise NotImplementedError


@system()
def one_epoch(
    message: OneEpochMessage,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    env_config: EnvConfig,
    training_state: TrainingState,
) -> None:
    device = env_config.device
    loss = model_config.loss
    model = model_config.model
    optimizer = env_config.optimizer
    scheduler = env_config.scheduler
    model.train()
    MessageWriter.info(
        f"\n{Color.ORANGE}{Color.BOLD} ------------------------------- Train  -------------------------------- {Color.END}\n"
    )
    with tqdm(
        dataset_config.train_loader,
        desc=f"Epoch {message.epoch}",
        leave=False,
        total=len(cast(TrainLoader, dataset_config.train_loader.dataset))
        * dataset_config.num_workers,
    ) as pbar:
        l_statistic = []
        for mask_input_mat, valid_len_mat, label_mat in pbar:

            mask_input_mat = mask_input_mat.to(device)
            valid_len_mat = valid_len_mat.to(device)

            optimizer.zero_grad()
            # [B, num_classes, S, H, W]
            pred_mat = model(mask_input_mat, None)
            l = loss(pred_mat, label_mat, valid_len_mat)
            l_statistic.append(l.cpu().item())
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_description(f"Loss: {np.mean(l_statistic):.5f}")

    scheduler.step()
    training_state.current_metrics = np.mean(l_statistic).item()
    report_str = f"Eval Loss: {training_state.current_metrics:.5f}"
    MessageWriter.info(report_str)
    # 最后10个 epoch 关闭数据增强
    if env_config.epochs - message.epoch <= 10:
        cast(TrainLoader, dataset_config.train_loader.dataset).toggle_enhance()


@system()
def eval_net(
    message: EvalMessage,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    env_config: EnvConfig,
    training_state: TrainingState,
) -> None:
    device = env_config.device
    model = model_config.model
    loss = model_config.loss
    model.eval()
    MessageWriter.info(
        f"\n{Color.BLUE}{Color.BOLD} ------------------------------- Test -------------------------------- {Color.END}\n"
    )
    with torch.no_grad():
        with tqdm(
            dataset_config.test_loader,
            desc=f"Eval {message.epoch}",
            leave=False,
            total=len(cast(TrainLoader, dataset_config.test_loader.dataset)),
        ) as pbar:
            l_statistic = []
            for mask_input_mat, valid_len_mat, label_mat in pbar:

                input_mat = mask_input_mat.to(device)
                valid_len_mat = valid_len_mat.to(device)
                label_mat = label_mat.to(device)

                # [B, num_classes, S, H, W]
                batch_size, num_step, num_rows, num_cols = input_mat.shape

                if (
                    num_rows < dataset_config.input_size
                    or num_cols < dataset_config.input_size
                ):
                    raise ValueError("The input size is too large.")

                pred_mat = torch.zeros(
                    (
                        batch_size,
                        dataset_config.voxel_params.max_z,
                        num_step,
                        num_rows,
                        num_cols,
                    ),
                    dtype=torch.int64,
                    device=input_mat.device,
                )

                for area_input_mat, area_valid_len_mat, pos in yield_input_mat(
                    input_mat, valid_len_mat, dataset_config.input_size
                ):
                    pred = model.refer(area_input_mat, area_valid_len_mat)
                    pred_mat[
                        :,
                        :,
                        : pred.shape[2],
                        pos[0] : pos[0] + dataset_config.input_size,
                        pos[1] : pos[1] + dataset_config.input_size,
                    ] = pred

                l = loss(pred_mat, label_mat, valid_len_mat)
                l_statistic.append(l.cpu().item())

        training_state.current_metrics = np.mean(l_statistic).item()
        report_str = f"Eval Loss: {training_state.current_metrics:.5f}"
        MessageWriter.info(report_str)


@system(message_type=StartTrainingMessage)
def start_training(env_config: EnvConfig, train_state: TrainingState) -> None:
    def callback(success: bool):
        if success:
            # 敲重点，重新发送该消息以开启下一个轮训练
            if train_state.current_epoch == env_config.epochs:
                return
            StartTrainingMessage.send()
        else:
            MessageWriter.error(
                RuntimeError(
                    f"\n{Color.RED}{Color.BOLD} ------------------------------- Epoch {train_state.current_epoch} Failed -------------------------------- {Color.END}\n"
                )
            )

    # Start 绿色 + 加粗
    MessageWriter.info(
        f"\n{Color.GREEN}{Color.BOLD} ------------------------------- Epoch {train_state.current_epoch} Start -------------------------------- {Color.END}\n"
    )

    with execute_block(
        group_id=f"epoch-{train_state.current_epoch}", callback=callback
    ):
        train_state.current_epoch += 1
        OneEpochMessage.send(train_state.current_epoch)
        EvalMessage.send(train_state.current_epoch)
        SaveCheckPointMessage.send()


def register_training_systems(app: ViridApp):
    app.register(training_lighting)
    app.register(start_training)

    app.register(save_checkpoint)
    app.register(load_checkpoint)
    app.register(load_mae_checkpoint)

    app.register(one_epoch)
    app.register(eval_net)
