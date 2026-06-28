"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import torch.nn as nn


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


import torch
import torch.nn as nn


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
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        # 这里的 apply 初始化只会初始化编码器内部的 Linear 层，非常安全
        self.apply(init_weights)

    def forward(self, seq_embedding, seq_mask):
        """
        参数:
            seq_embedding: [B * H * W, S, d_model]
            seq_mask:   [B * H * W, S]
        """

        # 直接应用 dropout 并送入 Transformer 骨干
        x = self.dropout(seq_embedding)
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

    def forward(self, spatial_feat, seq_feat, seq_mask):
        """
        参数:
            spatial_feat: 形状 [B * H * W, d_model]
            seq_feat:     形状 [B * H * W, S, d_model]
            seq_mask: 形状 [B * H * W, S] 的 BoolTensor
        """
        step_len = seq_feat.shape[1]
        spatial_feat_broadcasted = spatial_feat.unsqueeze(1).repeat(
            1, step_len, 1
        )  # [B * H * W, S, d_model]

        combined_feat = torch.cat([seq_feat, spatial_feat_broadcasted], dim=-1)
        fused_feat = self.feature_fusion(combined_feat)

        refined_feat = self.temporal_sequence_refiner(
            fused_feat, src_key_padding_mask=seq_mask
        )
        return refined_feat


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, d_model: int):
        super(TransformerClassifier, self).__init__()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, refined_feat):
        return self.classifier(refined_feat)
