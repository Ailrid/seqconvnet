"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from torch import nn
import torch
from ...structs import Tensor4D


def mat2seq(x):
    """将空间体素矩阵打扁为序列格式: [B, S, H, W] -> [B * H * W, S]"""
    num_steps = x.shape[1]
    return x.permute(0, 2, 3, 1).reshape(-1, num_steps)


class PositionalEncoding(nn.Module):
    """位置编码"""

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


def init_seq2seq(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])  # type: ignore


class RnnEncoder(nn.Module):
    """循环神经网络编码器"""

    def __init__(
        self,
        max_z: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout=0.1,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = PositionalEncoding(
            embed_size,
            max_z + 2,
        )
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.apply(init_seq2seq)

    def forward(self, input_mat: Tensor4D):
        # input_mat: [B, S, H, W]
        batch_size, _, num_rows, num_cols = input_mat.shape

        seq_mat = mat2seq(input_mat)  # [B * H * W, S]
        tokens = self.embedding(seq_mat.to(torch.int64)).permute(
            1, 0, 2
        )  # [S, B * H * W, embed_size]

        # state 形状: [num_layers, B * H * W, hidden_size]
        _, state = self.rnn(tokens)

        state = state.view(
            self.num_layers, batch_size, num_rows, num_cols, self.hidden_size
        )
        # 调整顺序为: [B, num_layers, hidden_size, H, W]
        state = state.permute(1, 0, 4, 2, 3).contiguous()
        # 平铺通道，融合成 4D 张量: [B, num_layers * hidden_size, H, W]
        state_mat = state.view(
            batch_size, self.num_layers * self.hidden_size, num_rows, num_cols
        )

        return state_mat


class RnnDecoder(nn.Module):
    """标准的、带步骤级上下文注入的循环神经网络解码器"""

    def __init__(
        self,
        num_classes: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(RnnDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_classes + 2, embed_size)

        # 核心修改：输入不仅是标签嵌入，还死死拼上了 Encoder 的上下文特征
        self.rnn = nn.GRU(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.apply(init_seq2seq)
        self.current_state = None

    def init_current_state(self, input_mat: Tensor4D):
        """初始化开始标签（BOS Token）"""
        batch_size, _, num_rows, num_cols = input_mat.shape
        self.current_state = torch.full(
            (batch_size * num_rows * num_cols, 1),
            fill_value=self.num_classes + 1,
            dtype=torch.int64,
            device=input_mat.device,
        )

    def next_step(self, current_token, gru_state, encoder_context):
        """
        推理阶段：自回归单步流转（带显式上下文拼接）
        参数:
            encoder_context: 对应当前空间格子的静态背景特征 -> 形状 [B * H * W, hidden_size]
        """
        # 标签 Embedding -> 形状 [1, B * H * W, embed_size]
        embed_input = self.embedding(current_token).permute(1, 0, 2)

        # 将静态空间上下文扩展并拼接到输入中
        # encoder_context.unsqueeze(0) 形状变为 [1, B * H * W, hidden_size]
        rnn_input = torch.cat([embed_input, encoder_context.unsqueeze(0)], dim=-1)

        # 驱动 GRU 推进，此时输入完全匹配 input_size = embed_size + hidden_size
        output, gru_state = self.rnn(
            rnn_input, gru_state
        )  # output: [1, B * H * W, hidden_size]

        return output, gru_state

    def forward(self, teach_mat, state_mat):
        """
        训练阶段：Teacher Forcing 并行训练（带时序级全量上下文拼接）
        """
        batch_size, seq_len, num_rows, num_cols = teach_mat.shape

        # 复原成 GRU 所需的初始隐状态 [num_layers, B * H * W, hidden_size]
        gru_initial_state = state_mat.view(
            batch_size, self.num_layers, self.hidden_size, num_rows, num_cols
        )
        gru_initial_state = gru_initial_state.permute(1, 0, 3, 4, 2).contiguous()
        gru_initial_state = gru_initial_state.view(
            self.num_layers, batch_size * num_rows * num_cols, self.hidden_size
        )

        # 提取静态空间上下文：采用 Encoder 最顶层的隐状态作为特征骨架
        encoder_context = gru_initial_state[-1]  # 形状: [B * H * W, hidden_size]

        # 标签 Embedding -> 形状 [S, B * H * W, embed_size]
        seq_mat = mat2seq(teach_mat)
        embed_output = self.embedding(seq_mat).permute(1, 0, 2)

        # 将上下文在时间轴（S 维）上复制，与全时序标签强行合体
        # context_expanded 形状: [S, B * H * W, hidden_size]
        context_expanded = encoder_context.unsqueeze(0).repeat(seq_len, 1, 1)

        # rnn_input 形状: [S, B * H * W, embed_size + hidden_size]
        rnn_input = torch.cat([embed_output, context_expanded], dim=-1)

        # 一口气送入 GRU 并行计算
        output, _ = self.rnn(
            rnn_input, gru_initial_state
        )  # output: [S, B * H * W, hidden_size]
        return output


class RnnClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super(RnnClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, output, teach_mat):
        batch_size, seq_len, num_rows, num_cols = teach_mat.shape
        #  [B, num_classes, S, H, W]
        logits = self.dense(output).permute(1, 0, 2)
        # [B * H * W, S, num_classes]
        logits = logits.view(batch_size, num_rows, num_cols, seq_len, self.num_classes)
        out = logits.permute(0, 4, 3, 1, 2).contiguous()

        return out  # [B,num_classes, S, H , W]
