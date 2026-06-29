"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

# 测试transformer版本是否能正常工作

from typing import cast
import warnings
import numpy as np
import torch
from tqdm import tqdm

from seqconvnet.core import (
    TrainLoader,
    TransformerShell,
    TransformerDecoder,
    TransformerEncoder,
    TransformerClassifier,
)
from seqconvnet.core import (
    SwinEncoder,
    VoxelParameters,
    SegmentationEvaluator,
    SoftDiceAndCELoss,
)
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from seqconvnet.core.nn.embedding import StandardHeightEmbedding

warnings.filterwarnings("ignore", message=".*nested_tensor.*")


def get_optimizer_and_scheduler(model, lr, warmup_epochs, epochs, weight_decay):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    # Warmup 阶段
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(epochs - warmup_epochs),
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return optimizer, scheduler


def get_model(
    input_size, num_classes, max_z, d_model, nhead, num_layers, dropout, device
):

    embedding = StandardHeightEmbedding(
        max_z,
        d_model,
    )
    # 组装网络
    seq_encoder = TransformerEncoder(
        d_model,
        nhead,
        num_layers,
        dropout,
    )

    conv_encoder = SwinEncoder(d_model, d_model, input_size)

    # conv_encoder = CustomConvEncoder(
    #     d_model,
    #     d_model,
    # )

    seq_decoder = TransformerDecoder(
        d_model,
        nhead,
        num_layers,
        dropout,
    )

    classifier = TransformerClassifier(
        num_classes,
        d_model,
    )

    model = TransformerShell(
        embedding, seq_encoder, conv_encoder, seq_decoder, classifier
    ).to(device)

    return model


def benchmark():
    input_size = 128
    max_z = 128
    d_model = 16
    nhead = 2
    num_layers = 2
    dropout = 0.1
    num_classes = 8
    device = "cuda"
    lr = 1e-3
    warmup_epochs = 5
    epochs = 200
    weight_decay = 1e-2

    print(f"Device: [{device.upper()}]")

    voxel_params = VoxelParameters(
        xy_resolution=0.5,
        z_resolution=0.5,
        max_z=max_z,
        min_rows=128,
        min_cols=128,
    )
    train_loader = DataLoader(
        TrainLoader(
            root_folder="preprocessed/dales_las/train",
            iter_times=1,
            input_size=128,
            voxel_params=voxel_params,
        )
    )
    evaluator = SegmentationEvaluator(num_classes, True)
    loss = SoftDiceAndCELoss(
        num_classes,
        [  # 每个类的类别权重
            1.0000,
            1.2866,
            3.2652,
            8.8057,
            7.3190,
            3.9489,
            8.2722,
            1.2322,
        ],
    )
    model = get_model(
        input_size, num_classes, max_z, d_model, nhead, num_layers, dropout, device
    )
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, lr, warmup_epochs, epochs, weight_decay
    )
    for i in range(epochs):
        with tqdm(
            train_loader,
            desc=f"Epoch {i}",
            leave=False,
            total=len(cast(TrainLoader, train_loader)),
        ) as pbar:
            l_statistic = []
            for input_mat, valid_len_mat, label_mat, _teach_mat in pbar:
                optimizer.zero_grad()
                # [B, num_classes + 1, S, H, W]
                pred_mat = model(input_mat, valid_len_mat)
                l = loss(pred_mat, label_mat, valid_len_mat)
                l_statistic.append(l.cpu().item())
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 计算精度,class从1开始，全部都+1
                pred_label = pred_mat.argmax(dim=1) + 1
                evaluator.update(pred_label, label_mat, valid_len_mat)

                pbar.set_description(f"Loss:{np.mean(l_statistic):.5f}")

        scheduler.step()
        evaluator.print_metrics()
        print(f"======= Epoch:{i + 1}, Loss:{np.mean(l_statistic):.5f} ======= ")


benchmark()
