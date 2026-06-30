"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class TransformerEncoder(nn.Module):

    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # 采用 Pre-LN 结构
        )

        # 显式定义最后一层的归一化组件
        final_norm = nn.LayerNorm(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=final_norm,
        )
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, seq_embedding, seq_mask):
        """参数:

        seq_embedding: [B * H * W, S, d_model]
        seq_mask:   [B * H * W, S]
        """
        x = self.dropout(seq_embedding)

        # 【已删除 checkpoint】还原为标准的 TransformerEncoder 前向调用
        seq_feat = self.transformer_encoder(x, src_key_padding_mask=seq_mask)

        # Masked Mean Pooling
        valid_mask = (~seq_mask).float().unsqueeze(-1)
        zeroed_feat = seq_feat * valid_mask

        sum_feat = torch.sum(zeroed_feat, dim=1)
        valid_counts = torch.sum(valid_mask, dim=1).clamp(min=1.0)
        pooled_feat = sum_feat / valid_counts

        return pooled_feat, seq_feat


class TransformerDecoder(nn.Module):

    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 💡 核心修改 1：换成标准的 TransformerDecoderLayer
        # 它内部包含两层注意力：第一层是高程特征的 Self-Attention，第二层是和空间特征的 Cross-Attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # 采用 Pre-LN 结构
        )

        final_norm = nn.LayerNorm(d_model)

        # 换成标准的 TransformerDecoder 容器
        self.temporal_sequence_refiner = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=final_norm,
        )
        # 假设 init_weights 函数在外部已定义
        self.apply(init_weights)

    def forward(self, spatial_feat, seq_feat, seq_mask):
        """参数:

        spatial_feat: 形状 [B * H * W, d_model] -> 空间平面特征
        seq_feat:     形状 [B * H * W, S, d_model] -> 高程序列特征
        seq_mask:     形状 [B * H * W, S] 的 BoolTensor -> Padding 掩码
        """
        # 通过 unsqueeze(1) 将空间特征的形状变为 [B * H * W, 1, d_model]
        # 这样它就变成了一个“序列长度为 1”的 Key/Value 记忆库（Memory）
        memory = spatial_feat.unsqueeze(1)

        # Query 就是你的高程时序特征
        tgt = seq_feat

        # 显式将 seq_mask 传给 tgt_key_padding_mask 以过滤高程的 padding
        refined_feat = self.temporal_sequence_refiner(
            tgt=tgt, memory=memory, tgt_key_padding_mask=seq_mask
        )

        return refined_feat  # 输出形状保持 [B * H * W, S, d_model]


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, d_model: int):
        super(TransformerClassifier, self).__init__()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, refined_feat):
        return self.classifier(refined_feat)
