"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
from ..interface import Network
from .rnn import RnnEncoder, RnnDecoder, RnnClassifier
from ...structs import Tensor4D


class RnnShell(Network):

    def __init__(
        self,
        seq_encoder: RnnEncoder,
        conv_encoder: torch.nn.Module,
        seq_decoder: RnnDecoder,
        classifier: RnnClassifier,
    ):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.conv_encoder = conv_encoder
        self.seq_decoder = seq_decoder
        self.classifier = classifier

    def forward(self, input_mat: Tensor4D, teach_mat: Tensor4D):
        """训练前向传播"""
        state = self.encode(input_mat)
        output = self.seq_decoder(teach_mat, state)
        return self.classifier(output, teach_mat)

    def encode(self, input_mat: Tensor4D):
        """联合特征编码：结合纵向 GRU 历史信息与 2D 卷积空间提纯特征"""
        state_mat = self.seq_encoder(input_mat)
        return state_mat + self.conv_encoder(state_mat)

    @torch.no_grad()
    def refer(self, input_mat: Tensor4D, valid_len_mat: Tensor4D):
        """
        配合标准 Seq2Seq 重构的自回归推理
        """
        batch_size, num_steps, num_rows, num_cols = input_mat.shape

        # 初始化 BOS 开始标签
        self.seq_decoder.init_current_state(input_mat)
        current_token = self.seq_decoder.current_state  # [B * H * W, 1]

        # 创建硬标签输出容器
        output = torch.zeros_like(input_mat, dtype=torch.int64)

        # 提取空间-时序混合特征
        init_state = self.encode(input_mat)

        # 解包复原为标准 GRU 初始隐状态
        num_layers = self.seq_decoder.num_layers
        hidden_size = self.seq_decoder.hidden_size

        gru_state = init_state.view(
            batch_size, num_layers, hidden_size, num_rows, num_cols
        )
        gru_state = gru_state.permute(1, 0, 3, 4, 2).contiguous()
        gru_state = gru_state.view(
            num_layers, batch_size * num_rows * num_cols, hidden_size
        )

        # 在进入自回归前，把 Encoder 的顶层空间特征作为“永恒的上下文”提取出来
        static_context = gru_state[-1].clone()  # 形状: [B * H * W, hidden_size]

        # 自回归状态机流转
        for i in range(num_steps - 1):
            # 把 static_context 塞进去
            output_feat, gru_state = self.seq_decoder.next_step(
                current_token, gru_state, static_context
            )
            logits_2d = self.classifier.dense(
                output_feat.squeeze(0)
            )  # [B * H * W, num_classes]
            # 恢复绝对类别,从1开始数
            current_token = logits_2d.argmax(dim=-1, keepdim=True) + 1
            # 塞入输出容器
            current_layer_label = current_token.reshape(
                batch_size, num_rows, num_cols, 1
            ).permute(0, 3, 1, 2)
            output[:, i : i + 1, :, :] = current_layer_label

        return output * valid_len_mat
