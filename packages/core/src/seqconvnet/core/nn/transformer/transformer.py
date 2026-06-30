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
        """
        参数:
            seq_embedding: [B * H * W, S, d_model]
            seq_mask:   [B * H * W, S]
        """
        x = self.dropout(seq_embedding)

        seq_feat = checkpoint(
            self.transformer_encoder,
            x,
            None,
            seq_mask,
            use_reentrant=False,
        )

        # Masked Mean Pooling
        valid_mask = (~seq_mask).float().unsqueeze(-1)
        zeroed_feat = seq_feat * valid_mask

        sum_feat = torch.sum(zeroed_feat, dim=1)
        valid_counts = torch.sum(valid_mask, dim=1).clamp(min=1.0)
        pooled_feat = sum_feat / valid_counts

        return pooled_feat, seq_feat


# class TransformerDecoder(nn.Module):
#     def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model

#         self.feature_fusion = nn.Sequential(
#             nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout)
#         )

#         decoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_model * 4,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,  # 采用 Pre-LN 结构
#         )

#         # 为解码时序提纯器显式定义最后一层的归一化组件
#         final_norm = nn.LayerNorm(d_model)

#         self.temporal_sequence_refiner = nn.TransformerEncoder(
#             decoder_layer,
#             num_layers=num_layers,
#             norm=final_norm,  # <--- 注入最终的 LayerNorm
#         )
#         self.apply(init_weights)

#     def forward(self, spatial_feat, seq_feat, seq_mask):
#         """
#         参数:
#             spatial_feat: 形状 [B * H * W, d_model]
#             seq_feat:     形状 [B * H * W, S, d_model]
#             seq_mask: 形状 [B * H * W, S] 的 BoolTensor
#         """
#         step_len = seq_feat.shape[1]
#         spatial_feat_broadcasted = spatial_feat.unsqueeze(1).repeat(
#             1, step_len, 1
#         )  # [B * H * W, S, d_model]

#         combined_feat = torch.cat([seq_feat, spatial_feat_broadcasted], dim=-1)
#         fused_feat = self.feature_fusion(combined_feat)

#         # 此时输出的 refined_feat 均值和方差重新回归正常分布，让分类器更好吃透特征
#         refined_feat = checkpoint(
#             self.temporal_sequence_refiner,
#             fused_feat,
#             None,
#             seq_mask,
#             use_reentrant=False,
#         )
#         return refined_feat


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
        """
        参数:
            spatial_feat: 形状 [B * H * W, d_model] -> 空间平面特征
            seq_feat:     形状 [B * H * W, S, d_model] -> 高程序列特征
            seq_mask:     形状 [B * H * W, S] 的 BoolTensor -> Padding 掩码
        """
        # 通过 unsqueeze(1) 将空间特征的形状变为 [B * H * W, 1, d_model]
        # 这样它就变成了一个“序列长度为 1”的 Key/Value 记忆库（Memory）
        memory = spatial_feat.unsqueeze(1)

        # Query 就是你的高程时序特征
        tgt = seq_feat

        refined_feat = checkpoint(
            self.temporal_sequence_refiner,
            tgt,  # 位置 1: tgt [B*H*W, S, d_model]
            memory,  # 位置 2: memory [B*H*W, 1, d_model]
            None,  # 位置 3: tgt_mask (非自回归并行预测，不需要下三角掩码)
            None,  # 位置 4: memory_mask
            seq_mask,  # 位置 5: tgt_key_padding_mask [B*H*W, S] (过滤高程 Padding)
            use_reentrant=False,
        )

        return refined_feat  # 输出形状保持 [B * H * W, S, d_model]


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, d_model: int):
        super(TransformerClassifier, self).__init__()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, refined_feat):
        return self.classifier(refined_feat)
