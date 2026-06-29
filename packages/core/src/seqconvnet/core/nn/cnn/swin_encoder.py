"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet (Refactored & Fixed)
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
        # 确保分辨率能被2整除
        assert H % 2 == 0 and W % 2 == 0, f"输入分辨率 ({H}x{W}) 无法被 2 整除"

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
    """将 [B, H, W, C] 转换为 [B * num_windows, window_size, window_size, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """将窗口特征还原回原本的图像形状 [B, H, W, C]"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """基于 PyTorch 原生 nn.MultiheadAttention 优化的窗口自注意力模块"""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

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

            # 🛠️ 核心修复 4：摒弃会产生实体内存复制的 .repeat()
            # 改用高级广播机制 .expand() 建立虚拟视图，在底层只进行一次 reshape，杜绝显存碎片化
            attn_mask = (
                mask.unsqueeze(0)
                .unsqueeze(2)
                .expand(num_repeats, -1, self.num_heads, -1, -1)
            )
            attn_mask = attn_mask.reshape(-1, N, N)

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
    🛠️ 核心修复 1 & 3：重构为符合官方标准的“单层”组件。
    通过外部传入的 shift_size 控制行为。内部自动校验低层分辨率，防止 Mask 退化失败。
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        # 🛠️ 核心修复 3：动态边界检查
        # 当输入特征图分辨率降到与窗口一样大时（例如Bottleneck的8x8），强制将位移设为0，
        # 使其完美退化为标准的全局窗口自注意力，防止错误的 Mask 把特征切碎。
        if self.input_resolution <= self.window_size:
            self.shift_size = 0
        else:
            self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

        # 只有在真正需要位移（SW-MSA）时，才离线计算并缓存注意力掩码
        if self.shift_size > 0:
            attn_mask = self._create_mask(
                input_resolution, window_size, self.shift_size
            )
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def _create_mask(self, res, win_size, shift_size):
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

        mask_windows = window_partition(img_mask, win_size)
        mask_windows = mask_windows.view(-1, win_size * win_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask != 0

    def forward(self, x):
        H, W = self.input_resolution, self.input_resolution
        C = self.dim

        shortcut = x
        x = self.norm1(x)

        # 🛠️ 核心修复 1：根据当前层的属性动态触发循环位移 (torch.roll)
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # 窗口切分与展平
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 计算自注意力（传入缓存的 self.attn_mask，若为标准 W-MSA 则是 None）
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 特征几何还原
        shifted_x = window_reverse(
            attn_windows.view(-1, self.window_size, self.window_size, C),
            self.window_size,
            H,
            W,
        )

        # 如果进行了前向循环位移，反向传播前必须等量 roll 回去对齐空间几何
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        # 残差连接与 FFN
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
            # nn.BatchNorm2d(c1),
            nn.GroupNorm(num_groups=16, num_channels=c1, eps=1e-6, affine=True),
            nn.GELU(),
        )

        # 引入可学习的绝对位置编码
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, img_size, img_size, c1))
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)


        # Stage 1: [128x128], dim=128, heads=4
        self.enc_stage1 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c1,
                    input_resolution=img_size,
                    num_heads=4,
                    window_size=8,
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c1,
                    input_resolution=img_size,
                    num_heads=4,
                    window_size=8,
                    shift_size=4,
                ),
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
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c2,
                    input_resolution=img_size // 2,
                    num_heads=8,
                    window_size=8,
                    shift_size=4,
                ),
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
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c3,
                    input_resolution=img_size // 4,
                    num_heads=16,
                    window_size=8,
                    shift_size=4,
                ),
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
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c4,
                    input_resolution=img_size // 8,
                    num_heads=32,
                    window_size=8,
                    shift_size=4,
                ),
            ]
        )
        self.down4 = PatchMerging(dim=c4)  # 16 -> 8

        # ==================== 【BOTTLENECK 最底层】 ====================
        # Bottleneck: [8x8], dim=2048, heads=64
        # 提示：在这里虽然写了 shift_size=4，但在初始化阶段会被核心修复3自动重置为0，完美确保全局无阻碍交互。
        self.bottleneck = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c5,
                    input_resolution=img_size // 16,
                    num_heads=64,
                    window_size=8,
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c5,
                    input_resolution=img_size // 16,
                    num_heads=64,
                    window_size=8,
                    shift_size=4,
                ),
            ]
        )

        # ==================== 【DECODER 阶段】 ====================
        # Stage 4 回弹: 8x8 -> 16x16
        self.up4 = PatchExpanding(input_dim=c5)  # 2048 -> 1024
        self.fusion4 = nn.Linear(c4 * 2, c4)
        self.dec_stage4 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c4,
                    input_resolution=img_size // 8,
                    num_heads=32,
                    window_size=8,
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c4,
                    input_resolution=img_size // 8,
                    num_heads=32,
                    window_size=8,
                    shift_size=4,
                ),
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
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c3,
                    input_resolution=img_size // 4,
                    num_heads=16,
                    window_size=8,
                    shift_size=4,
                ),
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
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c2,
                    input_resolution=img_size // 2,
                    num_heads=8,
                    window_size=8,
                    shift_size=4,
                ),
            ]
        )

        # Stage 1 回弹: 64x64 -> 128x128
        self.up1 = PatchExpanding(input_dim=c2)  # 256 -> 128
        self.fusion1 = nn.Linear(c1 * 2, c1)
        self.dec_stage1 = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=c1,
                    input_resolution=img_size,
                    num_heads=4,
                    window_size=8,
                    shift_size=0,
                ),
                SwinTransformerBlock(
                    dim=c1,
                    input_resolution=img_size,
                    num_heads=4,
                    window_size=8,
                    shift_size=4,
                ),
            ]
        )

        self.final_norm = nn.LayerNorm(c1)

    def _forward_blocks(self, blocks, x):
        """辅助函数：带有 checkpoint 检查的 Block 遍历执行"""
        for block in blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, x):
        # x: [B, 32, 128, 128]
        x = self.stem(x)  # [B, 128, 128, 128]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, 128, 128, 128]

        # 在骨干网络前叠加位置编码特征
        x = x + self.absolute_pos_embed

        # --------- ENCODER 递进与跳跃缓存 ---------
        x = self._forward_blocks(self.enc_stage1, x)
        skip1 = x  # 缓存 Stage 1 [B, 128, 128, 128]

        x = self.down1(x)  # -> [B, 64, 64, 256]
        x = self._forward_blocks(self.enc_stage2, x)
        skip2 = x  # 缓存 Stage 2 [B, 64, 64, 256]

        x = self.down2(x)  # -> [B, 32, 32, 512]
        x = self._forward_blocks(self.enc_stage3, x)
        skip3 = x  # 缓存 Stage 3 [B, 32, 32, 512]

        x = self.down3(
            x
        )  # 🛠️ 细节小瑕疵修正：注释修正为真实的通道数 -> [B, 16, 16, 1024]
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
        x = torch.cat([x, skip1], dim=-1)  # type: ignore # 拼接 -> [B, 128, 128, 256]
        x = self.fusion1(x)  # 降维 -> [B, 128, 128, 128]

        # 🛠️ 已修复：删除了多余的 "self.dec_stage1 =" 赋值
        x = self._forward_blocks(self.dec_stage1, x)

        # --------- 最终输出还原 ---------
        x = self.final_norm(x)
        out = x.permute(0, 3, 1, 2).contiguous()
        return out  # 最终输出: [B, 128, 128, 128]
