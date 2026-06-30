"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from torch import nn
import torch
from ...structs import Tensor3D
from torch.utils.checkpoint import checkpoint

def init_seq2seq(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])  # type: ignore


# class RnnEncoder(nn.Module):
#     """循环神经网络编码器（纯特征计算版）"""

#     def __init__(
#         self,
#         embed_size: int,
#         hidden_size: int,
#         num_layers: int,
#         dropout=0.1,
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size

#         self.rnn = nn.GRU(
#             input_size=embed_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#         )
#         self.apply(init_seq2seq)

#     def forward(self, embedding_seq: Tensor3D):
#         """
#         参数:
#             embedding_seq: [S, B * H * W, embed_size]
#         返回:
#             state: [num_layers, B * H * W, hidden_size]
#         """
#         _, state = self.rnn(embedding_seq)
#         return state


# class RnnDecoder(nn.Module):
#     """标准的、带步骤级上下文注入的循环神经网络解码器（外部注入查表策略）"""

#     def __init__(
#         self,
#         embed_size: int,
#         hidden_size: int,
#         num_layers: int,
#         dropout: float = 0.1,
#     ):
#         super(RnnDecoder, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size

#         # 输入：当前步的标签嵌入 (embed_size) + 空间上下文特征 (hidden_size)
#         self.rnn = nn.GRU(
#             input_size=embed_size + hidden_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#         )
#         self.apply(init_seq2seq)
#         self.current_state = None

#     def forward(self, embedding_teach, pooled_feat_seq):
#         """
#         参数:
#             embedding_teach: [S, B * H * W, embed_size] 外部准备好的全序列标签特征
#             pooled_feat_seq:  [num_layers, B * H * W, hidden_size] 编码层初始隐状态
#         """
#         encoder_context = pooled_feat_seq[
#             -1
#         ]  # 顶层静态空间上下文: [B * H * W, hidden_size]
#         seq_len = embedding_teach.shape[0]

#         # 沿时间步复制上下文: [S, B * H * W, hidden_size]
#         context_expanded = encoder_context.unsqueeze(0).repeat(seq_len, 1, 1)

#         # 历史特征与当前输入拼接
#         rnn_input = torch.cat([embedding_teach, context_expanded], dim=-1)
#         output, _ = self.rnn(
#             rnn_input, pooled_feat_seq
#         )  # output: [S, B * H * W, hidden_size]
#         return output


class RnnEncoder(nn.Module):
    """循环神经网络编码器（纯特征计算版 - 显存优化版）"""

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout=0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.apply(init_seq2seq)

    def forward(self, embedding_seq):
        """
        参数:
            embedding_seq: [S, B * H * W, embed_size]
        返回:
            state: [num_layers, B * H * W, hidden_size]
        """

        # 2. 将 GRU 的前向逻辑打包，丢弃其中间激活值
        def _inner_encoder(emb_seq):
            _, state = self.rnn(emb_seq)
            return state

        # 3. 训练模式下开启 checkpoint 省显存，验证模式下正常前向
        if self.training:
            return checkpoint(_inner_encoder, embedding_seq, use_reentrant=False)
        else:
            return _inner_encoder(embedding_seq)


class RnnDecoder(nn.Module):
    """标准的、带步骤级上下文注入的循环神经网络解码器（显存优化版）"""

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(RnnDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 输入：当前步的标签嵌入 (embed_size) + 空间上下文特征 (hidden_size)
        self.rnn = nn.GRU(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.apply(init_seq2seq)
        self.current_state = None

    def forward(self, embedding_teach, pooled_feat_seq):
        """
        参数:
            embedding_teach: [S, B * H * W, embed_size] 外部准备好的全序列标签特征
            pooled_feat_seq:  [num_layers, B * H * W, hidden_size] 编码层初始隐状态
        """

        # 4. 把涉及 repeat, cat, GRU 等吞噬显存的巨型中间体全部打包
        def _inner_decoder(emb_teach, feat_seq):
            encoder_context = feat_seq[
                -1
            ]  # 顶层静态空间上下文: [B * H * W, hidden_size]
            seq_len = emb_teach.shape[0]

            # 沿时间步复制上下文: [S, B * H * W, hidden_size]
            context_expanded = encoder_context.unsqueeze(0).repeat(seq_len, 1, 1)

            # 历史特征与当前输入拼接
            rnn_input = torch.cat([emb_teach, context_expanded], dim=-1)
            output, _ = self.rnn(
                rnn_input, feat_seq
            )  # output: [S, B * H * W, hidden_size]
            return output

        # 5. 同样只在训练时释放显存
        if self.training:
            return checkpoint(
                _inner_decoder, embedding_teach, pooled_feat_seq, use_reentrant=False
            )
        else:
            return _inner_decoder(embedding_teach, pooled_feat_seq)


class RnnClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super(RnnClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, output):
        # 输入 output: [S, B * H * W, hidden_size]
        # 返回 logits: [B * H * W, S, num_classes]
        return self.dense(output).permute(1, 0, 2)
