"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
from torch.nn import Module
from ...structs import Tensor4D
from .transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerClassifier,
    mat2seq,
)
from ..interface import Network


class TransformerShell(Network):

    def __init__(
        self,
        seq_encoder: TransformerEncoder,
        conv_encoder: torch.nn.Module,
        seq_decoder: TransformerDecoder,
        classifier: TransformerClassifier,
    ):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.conv_encoder = conv_encoder
        self.seq_decoder = seq_decoder
        self.classifier: Module = classifier
        # 显式记录 d_model，供后面 view 形状变换使用
        self.d_model = seq_encoder.d_model

    def forward(self, input_mat: Tensor4D, _teach_mat=None):
        batch_size, step_len, height, width = input_mat.shape

        # 扁平化输入
        seq_input = mat2seq(input_mat)  # [B * H * W, S]

        padding_mask = seq_input == 0  # [B * H * W, S]

        # 将 input_mat 修改为扁平化后的 seq_input 喂给编码器
        pooled_feat, seq_feat = self.seq_encoder(seq_input, padding_mask)

        # 转换成 2D 骨干网络需要的 4D 图像特征
        mat_feat = pooled_feat.view(batch_size, height, width, self.d_model)
        mat_feat = mat_feat.permute(0, 3, 1, 2).contiguous()  # [B, d_model, H, W]

        # 横向空间互看
        spatial_feat = self.conv_encoder(mat_feat)  # [B, d_model, H, W]

        # 【将空间地图特征精确解构并展平，保证空间索引对齐
        spatial_feat = spatial_feat.permute(0, 2, 3, 1).reshape(
            -1, self.d_model
        )  # [B * H * W, d_model]

        # 纵横特征交汇解码
        refined_feat = self.seq_decoder(spatial_feat, seq_feat, padding_mask)
        logits = self.classifier(refined_feat)  # [B * H * W, S, num_classes]

        # 折叠回五维预测形状
        logits = logits.view(batch_size, height, width, step_len, -1)
        out = logits.permute(0, 4, 3, 1, 2).contiguous()  # [B, num_classes, S, H, W]
        return out

    @torch.no_grad()
    def refer(self, input_mat: Tensor4D, valid_len_mat: Tensor4D):
        pred = self.forward(input_mat, valid_len_mat)
        output = pred.argmax(dim=1) + 1
        return output * valid_len_mat
