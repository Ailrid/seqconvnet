"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
from ..interface import Network
from .rnn import RnnEncoder, RnnDecoder, RnnClassifier
from ...structs import Tensor4D
from ..embedding import mat2seq


class RnnShell(Network):

    def __init__(
        self,
        embedding_encoder: torch.nn.Module,
        embedding_decoder: torch.nn.Module,
        seq_encoder: RnnEncoder,
        conv_encoder: torch.nn.Module,
        seq_decoder: RnnDecoder,
        classifier: RnnClassifier,
    ):
        super().__init__()
        self.embedding_encoder = embedding_encoder
        self.embedding_decoder = embedding_decoder
        self.seq_encoder = seq_encoder
        self.conv_encoder = conv_encoder
        self.seq_decoder = seq_decoder
        self.classifier = classifier

        self.num_layers = seq_encoder.num_layers
        self.hidden_size = seq_encoder.hidden_size
        # 稳妥抓取分类类别数，用于 5D 折叠
        self.num_classes = classifier.num_classes

    def forward(self, input_mat: Tensor4D, teach_mat: Tensor4D):
        """训练前向传播"""
        batch_size, step_len, num_rows, num_cols = input_mat.shape

        # pooled_feat_mat -> [B, num_layers * hidden_size, H, W]
        pooled_feat_mat = self.encode(input_mat)

        # pooled_fea_seq -> [num_layers, B * H * W, hidden_size]
        pooled_feat_seq = pooled_feat_mat.view(
            batch_size, self.num_layers, self.hidden_size, num_rows, num_cols
        )
        pooled_feat_seq = pooled_feat_seq.permute(1, 0, 3, 4, 2).contiguous()
        pooled_feat_seq = pooled_feat_seq.view(
            self.num_layers, batch_size * num_rows * num_cols, self.hidden_size
        )

        # 将标签矩阵打扁、查表、调整时序轴
        teach_seq = mat2seq(teach_mat).to(torch.int64)  # [B * H * W, S]
        embedding_teach = self.embedding_decoder(teach_seq).permute(
            1, 0, 2
        )  # [S, B * H * W, embed_size]

        # output -> [S, B * H * W, hidden_size]
        output = self.seq_decoder(embedding_teach, pooled_feat_seq)

        # logits -> [B * H * W, S, num_classes]
        logits = self.classifier(output)

        # 用 logits 还原空间对齐，并使用 self.num_classes
        out_mat = logits.view(
            batch_size, num_rows, num_cols, step_len, self.num_classes
        )
        out_mat = out_mat.permute(0, 4, 3, 1, 2).contiguous()
        return out_mat  # [B, num_classes, S, H, W]

    def encode(self, input_mat: Tensor4D):
        """混合编码管道"""
        batch_size, _, num_rows, num_cols = input_mat.shape

        seq_mat = mat2seq(input_mat).to(torch.int64)
        # [S, B * H * W, embed_size]
        embedding_seq = self.embedding_encoder(seq_mat).permute(1, 0, 2)

        # state_seq: [num_layers, B * H * W, hidden_size]
        state_seq = self.seq_encoder(embedding_seq)

        # 转换并组装空间 4D 特征
        state_mat = state_seq.view(
            self.num_layers, batch_size, num_rows, num_cols, self.hidden_size
        )
        state_mat = state_mat.permute(1, 0, 4, 2, 3).contiguous()
        state_mat = state_mat.view(
            batch_size, self.num_layers * self.hidden_size, num_rows, num_cols
        )

        return state_mat + self.conv_encoder(state_mat)

    @torch.no_grad()
    def refer(self, input_mat: Tensor4D, valid_len_mat: Tensor4D):
        """
        配合外置依赖注入的自回归闭环推理
        """
        batch_size, num_steps, num_rows, num_cols = input_mat.shape

        # [B, num_layers * hidden_size, H , W]
        pooled_feat_mat = self.encode(input_mat)

        # pooled_feat_seq [num_layers, B * H * W, hidden_size]
        pooled_feat_seq = pooled_feat_mat.view(
            batch_size, self.num_layers, self.hidden_size, num_rows, num_cols
        )
        pooled_feat_seq = pooled_feat_seq.permute(1, 0, 3, 4, 2).contiguous()
        gru_state = pooled_feat_seq.view(
            self.num_layers, batch_size * num_rows * num_cols, self.hidden_size
        )

        static_context = gru_state[-1].clone()  # [B * H * W, hidden_size]

        # BOS  -> [B * H * W, 1]
        current_token = torch.full(
            (batch_size * num_rows * num_cols, 1),
            fill_value=self.num_classes + 1,
            dtype=torch.int64,
            device=input_mat.device,
        )

        output = torch.zeros_like(input_mat, dtype=torch.int64)

        for i in range(num_steps - 1):
            # embed_input [1, B * H * W, embed_size]
            embed_input = self.embedding_decoder(current_token).permute(1, 0, 2)

            # context_expanded -> [1, B * H * W, hidden_size]
            context_expanded = static_context.unsqueeze(0)

            # rnn_input -> [1, B * H * W, embed_size + hidden_size]
            rnn_input = torch.cat([embed_input, context_expanded], dim=-1)

            output_feat, gru_state = self.seq_decoder.rnn(rnn_input, gru_state)

            logits_2d = self.classifier.dense(
                output_feat.squeeze(0)
            )  # [B * H * W, num_classes]

            # Argmax 抓取预测值（恢复绝对类别，从 1 开始数）
            current_token = logits_2d.argmax(dim=-1, keepdim=True) + 1

            # 将当前步的硬标签塞回空间 4D 对应的时间图层切片
            current_layer_label = current_token.reshape(
                batch_size, num_rows, num_cols, 1
            ).permute(0, 3, 1, 2)

            output[:, i : i + 1, :, :] = current_layer_label

        return output * valid_len_mat
