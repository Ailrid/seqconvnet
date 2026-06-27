"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import torch.nn as nn


def mat2seq(x):
    num_steps = x.shape[1]
    return x.permute(0, 2, 3, 1).reshape(-1, num_steps)


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


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((max_len, embed_size))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, embed_size, 2, dtype=torch.float32) / embed_size
        )
        self.P[:, 0::2] = torch.sin(X)
        self.P[:, 1::2] = torch.cos(X)
        self.P = nn.Parameter(self.P, requires_grad=False)

    def forward(self, x):
        return self.P[x]


class TransformerEncoder(nn.Module):
    def __init__(self, max_z=64, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        num_embeddings = max_z + 2
        self.absolute_height_embedding = PositionalEncoding(
            embed_size=d_model, max_len=num_embeddings
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, seq_input, padding_mask):
        """
        参数:
            padding_mask: 形状 [B * H * W, S] 的 BoolTensor, True 代表是 Padding 区域
        """
        x = self.absolute_height_embedding(seq_input)  # [B * H * W, S, d_model]
        x = self.dropout(x)

        # 传入标准的布尔 key_padding_mask
        seq_feat = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Masked Mean Pooling 改造
        # 将布尔 padding_mask 反转并转为 float，得到 1代表有效、0代表Padding 的矩阵
        valid_mask = (~padding_mask).float().unsqueeze(-1)  # [B * H * W, S, 1]
        zeroed_feat = seq_feat * valid_mask

        sum_feat = torch.sum(zeroed_feat, dim=1)
        valid_counts = torch.sum(valid_mask, dim=1).clamp(min=1.0)
        pooled_feat = sum_feat / valid_counts  # [B * H * W, d_model]

        return pooled_feat, seq_feat


class TransformerDecoder(nn.Module):
    def __init__(self, num_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_sequence_refiner = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )
        self.apply(init_weights)

    def forward(self, spatial_feat, seq_feat, padding_mask):
        """
        参数:
            spatial_feat: 形状 [B * H * W, d_model]
            seq_feat:     形状 [B * H * W, S, d_model]
            padding_mask: 形状 [B * H * W, S] 的 BoolTensor
        """
        step_len = seq_feat.shape[1]
        spatial_feat_broadcasted = spatial_feat.unsqueeze(1).repeat(
            1, step_len, 1
        )  # [B * H * W, S, d_model]

        combined_feat = torch.cat([seq_feat, spatial_feat_broadcasted], dim=-1)
        fused_feat = self.feature_fusion(combined_feat)

        # 直接使用标准的 padding_mask，不需要执行危险的 float 取反
        refined_feat = self.temporal_sequence_refiner(
            fused_feat, src_key_padding_mask=padding_mask
        )
        return refined_feat


class TransformerClassifier(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super(TransformerClassifier, self).__init__()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, refined_feat):
        return self.classifier(refined_feat)
