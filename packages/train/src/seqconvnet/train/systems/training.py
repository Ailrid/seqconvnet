"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import numpy as np
from typing import cast
import torch

from ..messages.training import (
    StartTrainingMessage,
    CheckPointMessage,
    OneEpochMessage,
    EvalMessage,
)
from ..components import ModelConfig, EnvConfig, DatasetConfig, TrainingState
from virid.core import system, ViridApp, MessageWriter
from virid.std import execute_block
from tqdm import tqdm
from seqconvnet.core import TrainLoader, refer_mat


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
def one_epoch(
    message: OneEpochMessage,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    env_config: EnvConfig,
) -> None:

    evaluator = env_config.evaluator
    evaluator.reset()
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
        total=len(cast(TrainLoader, dataset_config.train_loader.dataset)),
    ) as pbar:
        l_statistic = []
        for input_mat, valid_len_mat, label_mat, _teach_mat in pbar:
            optimizer.zero_grad()
            # [B, num_classes, S, H, W]
            pred_mat = model(input_mat, valid_len_mat)
            l = loss(pred_mat, label_mat, valid_len_mat)
            l_statistic.append(l.cpu().item())
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 计算精度,class从1开始，全部都+1
            pred_label = pred_mat.argmax(dim=1) + 1
            evaluator.update(pred_label, label_mat, valid_len_mat)
            pbar.set_description(f"Loss: {np.mean(l_statistic):.5f}")

    scheduler.step()
    _, _, report_str = evaluator.print_metrics()
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
    evaluator = env_config.evaluator
    evaluator.reset()
    model = model_config.model
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
            for input_mat, valid_len_mat, label_mat in pbar:
                # [B, num_classes, S, H, W]
                pred_label = refer_mat(
                    input_mat,
                    valid_len_mat,
                    model,
                    dataset_config.input_size,
                )
                evaluator.update(pred_label, label_mat, valid_len_mat)

        hist_matrix, metrics, report_str = evaluator.print_metrics()
        training_state.hist_matrix = hist_matrix
        training_state.metrics = metrics
        MessageWriter.info(report_str)


@system()
def checkpoint(
    message: CheckPointMessage,
) -> None:
    pass


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
        CheckPointMessage.send()


def register_training_systems(app: ViridApp):
    app.register(one_epoch)
    app.register(eval_net)
    app.register(checkpoint)
    app.register(start_training)
