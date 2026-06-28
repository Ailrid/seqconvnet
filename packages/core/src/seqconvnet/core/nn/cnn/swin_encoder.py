"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class PatchMerging(nn.Module):
    """Swin下采样：分辨率减半，通道数翻倍 [B, H, W, C] ---> [B, H/2, W/2, 2C]"""

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        return self.reduction(x)


class PatchExpanding(nn.Module):
    """Swin上采样：分辨率加倍，通道数减半 [B, H, W, C] ---> [B, 2H, 2W, C/2]"""

    def __init__(self, input_dim):
        super().__init__()
        self.expand = nn.Linear(input_dim, 2 * input_dim, bias=False)
        self.norm = nn.LayerNorm(input_dim // 2)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = x.view(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 4)
        return self.norm(x)


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
    4阶深层主网络：支持 128 -> 64 -> 32 -> 16 -> 8 的全对称U型特征感应器
    """

    def __init__(
        self, in_channels=32, out_channels=128, img_size=128, use_checkpoint=True
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # 动态计算 5 个层级的通道数
        c1 = out_channels  # Stage 1: 128
        c2 = out_channels * 2  # Stage 2: 256
        c3 = out_channels * 4  # Stage 3: 512
        c4 = out_channels * 8  # Stage 4: 1024
        c5 = out_channels * 16  # Bottleneck: 2048

        # 0. Stem 映射层 (128x128)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )

        # ==================== 【ENCODER 阶段】 ====================
        # Stage 1: [128x128], dim=128, heads=4
        self.enc_stage1 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c1, input_resolution=img_size, num_heads=4, window_size=8
                )
                for _ in range(2)
            ]
        )
        self.down1 = PatchMerging(dim=c1)  # 128 -> 64

        # Stage 2: [64x64], dim=256, heads=8
        self.enc_stage2 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c2,
                    input_resolution=img_size // 2,
                    num_heads=8,
                    window_size=8,
                )
                for _ in range(2)
            ]
        )
        self.down2 = PatchMerging(dim=c2)  # 64 -> 32

        # Stage 3: [32x32], dim=512, heads=16
        self.enc_stage3 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c3,
                    input_resolution=img_size // 4,
                    num_heads=16,
                    window_size=8,
                )
                for _ in range(2)
            ]
        )
        self.down3 = PatchMerging(dim=c3)  # 32 -> 16

        # Stage 4: [16x16], dim=1024, heads=32
        self.enc_stage4 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c4,
                    input_resolution=img_size // 8,
                    num_heads=32,
                    window_size=8,
                )
                for _ in range(2)
            ]
        )
        self.down4 = PatchMerging(dim=c4)  # 16 -> 8

        # ==================== 【BOTTLENECK 最底层】 ====================
        # Bottleneck: [8x8], dim=2048, heads=64
        # 此时特征图大小与窗口大小一致(8x8)，滑动窗口会退化为标准全局/窗口交互，完美闭环
        self.bottleneck = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c5,
                    input_resolution=img_size // 16,
                    num_heads=64,
                    window_size=8,
                )
                for _ in range(2)
            ]
        )

        # ==================== 【DECODER 阶段】 ====================
        # Stage 4 回弹: 8x8 -> 16x16
        self.up4 = PatchExpanding(input_dim=c5)  # 2048 -> 1024
        self.fusion4 = nn.Linear(c4 * 2, c4)  # 融合层：把skip连接的1024+1024映射回1024
        self.dec_stage4 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c4,
                    input_resolution=img_size // 8,
                    num_heads=32,
                    window_size=8,
                )
                for _ in range(2)
            ]
        )

        # Stage 3 回弹: 16x16 -> 32x32
        self.up3 = PatchExpanding(input_dim=c4)  # 1024 -> 512
        self.fusion3 = nn.Linear(c3 * 2, c3)
        self.dec_stage3 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c3,
                    input_resolution=img_size // 4,
                    num_heads=16,
                    window_size=8,
                )
                for _ in range(2)
            ]
        )

        # Stage 2 回弹: 32x32 -> 64x64
        self.up2 = PatchExpanding(input_dim=c3)  # 512 -> 256
        self.fusion2 = nn.Linear(c2 * 2, c2)
        self.dec_stage2 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c2,
                    input_resolution=img_size // 2,
                    num_heads=8,
                    window_size=8,
                )
                for _ in range(2)
            ]
        )

        # Stage 1 回弹: 64x64 -> 128x128
        self.up1 = PatchExpanding(input_dim=c2)  # 256 -> 128
        self.fusion1 = nn.Linear(c1 * 2, c1)
        self.dec_stage1 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c1, input_resolution=img_size, num_heads=4, window_size=8
                )
                for _ in range(2)
            ]
        )

        self.final_norm = nn.LayerNorm(c1)

    def _forward_blocks(self, blocks, x):
        """辅助函数：带有 checkpoint 检查的 Block 遍历执行"""
        for block in blocks:
            if self.use_checkpoint and self.training:
                # 采用稳定的 use_reentrant=False 机制
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, x):
        # 输入 x: [B, 32, 128, 128]
        x = self.stem(x)  # [B, 128, 128, 128]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, 128, 128, 128]

        # --------- ENCODER 递进与跳跃缓存 ---------
        x = self._forward_blocks(self.enc_stage1, x)
        skip1 = x  # 缓存 Stage 1 [B, 128, 128, 128]

        x = self.down1(x)  # -> [B, 64, 64, 256]
        x = self._forward_blocks(self.enc_stage2, x)
        skip2 = x  # 缓存 Stage 2 [B, 64, 64, 256]

        x = self.down2(x)  # -> [B, 32, 32, 512]
        x = self._forward_blocks(self.enc_stage3, x)
        skip3 = x  # 缓存 Stage 3 [B, 32, 32, 512]

        x = self.down3(x)  # -> [B, 16, 16, 1042]
        x = self._forward_blocks(self.enc_stage4, x)
        skip4 = x  # 缓存 Stage 4 [B, 16, 16, 1024]

        x = self.down4(x)  # -> [B, 8, 8, 2048]

        # --------- BOTTLENECK 最底层语义凝练 ---------
        x = self._forward_blocks(self.bottleneck, x)  # [B, 8, 8, 2048]

        # --------- DECODER 层层回弹与 U 型融合 ---------
        # 融合 Stage 4
        x = self.up4(x)  # 上采样 -> [B, 16, 16, 1024]
        x = torch.cat([x, skip4], dim=-1)  # type: ignore # 拼接 -> [B, 16, 16, 2048]
        x = self.fusion4(x)  # 降维 -> [B, 16, 16, 1024]
        x = self._forward_blocks(self.dec_stage4, x)

        # 融合 Stage 3
        x = self.up3(x)  # 上采样 -> [B, 32, 32, 512]
        x = torch.cat([x, skip3], dim=-1)  # type: ignore # 拼接 -> [B, 32, 32, 1024]
        x = self.fusion3(x)  # 降维 -> [B, 32, 32, 512]
        x = self._forward_blocks(self.dec_stage3, x)

        # 融合 Stage 2
        x = self.up2(x)  # 上采样 -> [B, 64, 64, 256]
        x = torch.cat([x, skip2], dim=-1)  # type: ignore  # 拼接 -> [B, 64, 64, 512]
        x = self.fusion2(x)  # 降维 -> [B, 64, 64, 256]
        x = self._forward_blocks(self.dec_stage2, x)

        # 融合 Stage 1
        x = self.up1(x)  # 上采样 -> [B, 128, 128, 128]
        x = torch.cat([x, skip1], dim=-1)  # type: ignore  # 拼接 -> [B, 128, 128, 256]
        x = self.fusion1(x)  # 降维 -> [B, 128, 128, 128]
        x = self._forward_blocks(self.dec_stage1, x)

        # --------- 最终输出还原 ---------
        x = self.final_norm(x)
        out = x.permute(0, 3, 1, 2).contiguous()
        return out  # 最终输出: [B, 128, 128, 128]
