"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import torch.nn as nn


def window_partition(x, window_size=8):
    """
    将 [B, H, W, C] 转换为 [B * num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    # 划分成一个个标准小窗口
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 排列组合并打扁成 Batch 维度
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口特征还原回原本的图像形状 [B, H, W, C]
    """
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    基于 PyTorch 原生 nn.MultiheadAttention 优化的窗口自注意力模块
    """

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # 规范化改进：使用 PyTorch 工业级原生多头注意力
        # batch_first=True 确保输入输出形状规整为 [B*, N, C]，内部自动集成 QKV 投影与 Out 映射层
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, bias=True
        )

    def forward(self, x, mask=None):
        """
        x: [B*num_windows, N, C], 其中 N = window_size * window_size
        mask: [num_windows, N, N] 滑动窗口边缘掩码 (BoolTensor)
        """
        B_, N, C = x.shape

        attn_mask = None
        if mask is not None:
            nW = mask.shape[0]
            num_repeats = B_ // nW

            # 适配 PyTorch 官方标配的注意力掩码形状 [B_ * num_heads, N, N]
            # 这里的 mask 已经是初始化算好的 BoolTensor (True 表示屏蔽，False 表示保留)
            attn_mask = mask.repeat(num_repeats, 1, 1)  # [B_, N, N]
            attn_mask = attn_mask.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1
            )  # [B_, num_heads, N, N]
            attn_mask = attn_mask.view(-1, N, N)  # [B_ * num_heads, N, N]

        # 将 x 同时作为 Query, Key, Value 传入
        # self.attn 返回一个元组 (output_tensor, weights_tensor)，我们通过 [0] 只取特征输出
        attn_output, _ = self.attn(query=x, key=x, value=x, attn_mask=attn_mask)
        return attn_output


class Mlp(nn.Module):
    """标准的 Transformer 前馈网络 (FFN)"""

    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    单个 Swin Transformer 块：包含连续的 W-MSA（标准窗口）与 SW-MSA（滑动窗口）
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2  # 滑动步长为窗口大小的一半 (4)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

        # 在模型初始化时，提前离线计算好滑动窗口的布尔掩码矩阵，节省显存与在线算力
        self.register_buffer(
            "attn_mask",
            self._create_mask(input_resolution, window_size, self.shift_size),
        )

    def _create_mask(self, res, win_size, shift_size):
        # 创建一个和图像大小一致的辅助矩阵，用来给不同切片区域打上独一无二的标签数字
        img_mask = torch.zeros((1, res, res, 1))
        h_slices = (
            slice(0, -win_size),
            slice(-win_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -win_size),
            slice(-win_size, -shift_size),
            slice(-shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 划分窗口并打扁
        mask_windows = window_partition(
            img_mask, win_size
        )  # [nW, win_size, win_size, 1]
        mask_windows = mask_windows.view(-1, win_size * win_size)

        # 利用广播机制做差，差值不为 0 的地方说明跨越了不该交互的物理边界
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        # 规范化改进：直接返回布尔矩阵（True 代表跨边界需要‘消音’，False 代表可以正常交互）
        # 这把计算开销直接压缩到了初始化阶段
        return attn_mask != 0

    def forward(self, x):
        # x 形状: [B, H, W, C]
        H, W = self.input_resolution, self.input_resolution
        C = self.dim

        # ------------------ 第一阶段：标准局部窗口自注意力 (W-MSA) ------------------
        shortcut = x
        x = self.norm1(x)

        # 窗口切分与展平
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 计算注意力（标准窗口不需要 mask）
        attn_windows = self.attn(x_windows, mask=None)

        # 特征还原与残差连接
        x = window_reverse(
            attn_windows.view(-1, self.window_size, self.window_size, C),
            self.window_size,
            H,
            W,
        )
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        # ------------------ 第二阶段：移位/滑动窗口自注意力 (SW-MSA) ------------------
        shortcut = x
        x = self.norm1(x)

        # 核心：使用 torch.roll 算子在空间分辨率上进行循环位移
        shifted_x = torch.roll(
            x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
        )

        # 再次切分窗口并展平
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 计算滑动注意力（注入模型初始化时存好的边缘布尔掩码）
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 还原窗口特征
        shifted_x = window_reverse(
            attn_windows.view(-1, self.window_size, self.window_size, C),
            self.window_size,
            H,
            W,
        )

        # 核心反向还原：沿反方向 roll 回去，把特征在空间几何位置上完全对齐
        x = torch.roll(
            shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
        )
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class SwinEncoder(nn.Module):
    """
    点云 BEV 特征图的高性能横向空间感应器
    """

    def __init__(self, in_channels=32, out_channels=128, img_size=128):
        super().__init__()

        # 特征图 Stem：保持点云无损分辨率 (Stride=1)，无缝转换通道
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # 使用 ModuleList 规范化堆叠 Block，有利于维护和清晰的管理
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=out_channels,
                    input_resolution=img_size,
                    num_heads=4,
                    window_size=8,
                ),
                SwinTransformerBlock(
                    dim=out_channels,
                    input_resolution=img_size,
                    num_heads=4,
                    window_size=8,
                ),
            ]
        )

        self.final_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # 输入 x: [B, 32, 128, 128]
        x = self.stem(x)  # [B, 128, 128, 128]

        # 转换至 Transformer 要求的通道末尾格式: [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # 遍历执行空间特征交融
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # 还原回 PyTorch 传统的 CNN 通道优先格式: [B, C, H, W]
        out = x.permute(0, 3, 1, 2).contiguous()
        return out  # 输出 out: [B, 128, 128, 128]
