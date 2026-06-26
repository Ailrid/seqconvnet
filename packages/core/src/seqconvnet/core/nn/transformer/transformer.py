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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        max_z=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        num_embeddings = max_z + 2
        self.absolute_height_embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=d_model, padding_idx=0
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

    def forward(self, input_mat, valid_len_mat):
        batch_size, _step_len, height, width = input_mat.shape

        # 扁平化
        seq_input = mat2seq(input_mat)
        seq_mask = mat2seq(valid_len_mat)

        # 绝对高度查表与特征提取
        x = self.absolute_height_embedding(seq_input)
        x = self.dropout(x)

        padding_mask = ~seq_mask.to(torch.bool)
        # 纵向 Transformer 编码 -> 输出形状: [B * H * W, S, d_model]
        seq_feat = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # 将 mask 为 0 (填充位) 的特征赋予极小值，防止其参与 max-pooling 后干扰正常特征
        # seq_mask.unsqueeze(-1) 形状变为 [B * H * W, S, 1]，利用广播自动对齐 d_model 维度
        output = seq_feat.masked_fill(seq_mask.unsqueeze(-1) == 0, float("-1e9"))

        # 全局最大池化,形状变为: [B * H * W, d_model]
        pooled_feat = torch.max(output, dim=1)[0]

        # [B * H * W, d_model] -> [B, H, W, d_model]
        mat_feat = pooled_feat.view(batch_size, height, width, self.d_model)

        # 调整通道顺序 [B, H, W, d_model] -> [B, d_model, H, W]
        mat_feat = mat_feat.permute(0, 3, 1, 2).contiguous()

        return (mat_feat, seq_feat)


class TransformerDecoder(nn.Module):
    """
    非自回归的序列特征解码器：
    保持函数接口不变，但用 Transformer 块替代 MLP，实现类似第二个 GRU 的纵向序列特征演进。
    """

    def __init__(self, num_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # 1. 空间特征与高度特征拼接后的初始降维融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout)
        )

        # 🟢 核心升级：引入标准 Transformer Encoder 作为一个序列精炼器
        # 它的任务是在融合空间特征后，替代你以前的“第二个 GRU”，在 Z 轴（S 维度）上做纵向全序列交互
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # 确保输入形状识别为 [Batch, Sequence, Channel]
            norm_first=True,
        )
        self.temporal_sequence_refiner = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )

    def forward(self, seq_feat, mat_feat, valid_len_mat):
        """
        参数:
            seq_feat:      Encoder 出来的纵向高度特征 -> 形状 [B * H * W, S, d_model]
            mat_feat:      2D 骨干网络提取的空间特征  -> 形状 [B, d_model, H, W]
            valid_len_mat: 原始的 mask 矩阵 (1有效, 0填充) -> 形状 [B, S, H, W]
        """
        batch_size, step_len, height, width = valid_len_mat.shape

        # 空间特征平铺并广播到每个高度层
        spatial_feat = mat2seq(mat_feat)  # [B * H * W, d_model]
        spatial_feat_broadcasted = spatial_feat.unsqueeze(1).repeat(
            1, step_len, 1
        )  # [B * H * W, S, d_model]

        # 空间与高度特征融合
        combined_feat = torch.cat(
            [seq_feat, spatial_feat_broadcasted], dim=-1
        )  # [B * H * W, S, d_model * 2]
        fused_feat = self.feature_fusion(combined_feat)  # [B * H * W, S, d_model]

        # 全序列自注意力计算
        # 将原先 [B, S, H, W] 的有效长度矩阵打扁为 [B * H * W, S]
        seq_mask = mat2seq(valid_len_mat)
        # 在 PyTorch Transformer 中，src_key_padding_mask 里面 True 代表要被抹消的填充位
        key_padding_mask = seq_mask == 0

        # 让特征在 Z 轴上流动起来，同时利用掩码隔绝无效层
        refined_feat = self.temporal_sequence_refiner(
            fused_feat, src_key_padding_mask=key_padding_mask
        )  # [B * H * W, S, d_model]
        return refined_feat


class TransformerClassifier(nn.Module):

    def __init__(self, d_model: int, num_classes: int):
        super(TransformerClassifier, self).__init__()
        # 最终的体素分类器
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, refined_feat, valid_len_mat):
        batch_size, step_len, height, width = valid_len_mat.shape
        # 输出预测
        logits = self.classifier(refined_feat)  # [B * H * W, S, num_classes + 1]

        # 恢复成原始的 5D 空间体素形状
        logits = logits.view(batch_size, height, width, step_len, -1)
        out = logits.permute(0, 4, 3, 1, 2).contiguous()

        # 把原先是 0 (填充位) 的体素预测概率强制清零
        # out = out.masked_fill(valid_len_mat.unsqueeze(1) == 0, -1e9)
        return out  # [B ,num_classes + 1 ,S, W, H]
