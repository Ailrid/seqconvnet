"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    """重叠图像块嵌入：用于下采样并提取局部空间特征"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            bias=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # 转换为 Transformer 要求的 [B, Token数量, 通道数] 格式
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixFFN(nn.Module):
    """混合前馈网络：利用 3x3 深度卷积替代位置编码，非常适合小图"""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        # 深度卷积 (Depth-wise Convolution) 注入局部上下文
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x = self.fc1(x)

        # 转回二维张量做卷积
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.act(x)
        x = self.fc2(x)
        return x


class EfficientSelfAttention(nn.Module):
    """高效自注意力：利用还原率 R 压缩 K、V 的空间维度，大幅度提升小编码器的 FPS"""

    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).view(B, C, H, W)
            x_ = self.sr(x_).flatten(2).transpose(1, 2)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, N, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """SegFormer 的标准 Transformer Block"""

    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class CustomSegFormerB0(nn.Module):
    def __init__(self, in_channels=32, out_channels=1):
        super().__init__()

        # --- 骨干网络配置 (MiT-B0) ---
        dims = [32, 64, 160, 256]
        heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]

        # Stage 1 (输入 128x128 -> 变成 32x32)
        self.patch_embed1 = OverlapPatchEmbed(
            patch_size=7, stride=4, in_chans=in_channels, embed_dim=dims[0]
        )
        self.block1 = nn.ModuleList(
            [
                TransformerBlock(dim=dims[0], num_heads=heads[0], sr_ratio=sr_ratios[0])
                for _ in range(2)
            ]
        )
        self.norm1 = nn.LayerNorm(dims[0])

        # Stage 2 (32x32 -> 变成 16x16)
        self.patch_embed2 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=dims[0], embed_dim=dims[1]
        )
        self.block2 = nn.ModuleList(
            [
                TransformerBlock(dim=dims[1], num_heads=heads[1], sr_ratio=sr_ratios[1])
                for _ in range(2)
            ]
        )
        self.norm2 = nn.LayerNorm(dims[1])

        # Stage 3 (16x16 -> 变成 8x8)
        self.patch_embed3 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=dims[1], embed_dim=dims[2]
        )
        self.block3 = nn.ModuleList(
            [
                TransformerBlock(dim=dims[2], num_heads=heads[2], sr_ratio=sr_ratios[2])
                for _ in range(2)
            ]
        )
        self.norm3 = nn.LayerNorm(dims[2])

        # Stage 4 (8x8 -> 变成 4x4)
        self.patch_embed4 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=dims[2], embed_dim=dims[3]
        )
        self.block4 = nn.ModuleList(
            [
                TransformerBlock(dim=dims[3], num_heads=heads[3], sr_ratio=sr_ratios[3])
                for _ in range(2)
            ]
        )
        self.norm4 = nn.LayerNorm(dims[3])

        # --- 全 MLP 解码器配置 ---
        decoder_dim = 256
        self.linear_c4 = nn.Linear(dims[3], decoder_dim)
        self.linear_c3 = nn.Linear(dims[2], decoder_dim)
        self.linear_c2 = nn.Linear(dims[1], decoder_dim)
        self.linear_c1 = nn.Linear(dims[0], decoder_dim)

        # 融合 4 个 Stage 的特征
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

        # 最终映射层（不要分类，直接回归输出指定的通道数）
        self.dropout = nn.Dropout(0.1)
        self.predictor = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape

        # ---- Stage 1 ----
        x, h1, w1 = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h1, w1)
        c1 = self.norm1(x).view(B, h1, w1, -1).permute(0, 3, 1, 2)  # [B, 32, 32, 32]

        # ---- Stage 2 ----
        x, h2, w2 = self.patch_embed2(c1)
        for blk in self.block2:
            x = blk(x, h2, w2)
        c2 = self.norm2(x).view(B, h2, w2, -1).permute(0, 3, 1, 2)  # [B, 64, 16, 16]

        # ---- Stage 3 ----
        x, h3, w3 = self.patch_embed3(c2)
        for blk in self.block3:
            x = blk(x, h3, w3)
        c3 = self.norm3(x).view(B, h3, w3, -1).permute(0, 3, 1, 2)  # [B, 160, 8, 8]

        # ---- Stage 4 ----
        x, h4, w4 = self.patch_embed4(c3)
        for blk in self.block4:
            x = blk(x, h4, w4)
        c4 = self.norm4(x).view(B, h4, w4, -1).permute(0, 3, 1, 2)  # [B, 256, 4, 4]

        # ---- 全 MLP 解码器 (Decoder) ----
        # 1. 统一转换各层通道，并统一上采样到 Stage 1 的尺寸 (32x32)
        _c4 = (
            self.linear_c4(c4.flatten(2).transpose(1, 2))
            .transpose(1, 2)
            .view(B, -1, h4, w4)
        )
        _c4 = F.interpolate(_c4, size=(h1, w1), mode="bilinear", align_corners=False)

        _c3 = (
            self.linear_c3(c3.flatten(2).transpose(1, 2))
            .transpose(1, 2)
            .view(B, -1, h3, w3)
        )
        _c3 = F.interpolate(_c3, size=(h1, w1), mode="bilinear", align_corners=False)

        _c2 = (
            self.linear_c2(c2.flatten(2).transpose(1, 2))
            .transpose(1, 2)
            .view(B, -1, h2, w2)
        )
        _c2 = F.interpolate(_c2, size=(h1, w1), mode="bilinear", align_corners=False)

        _c1 = (
            self.linear_c1(c1.flatten(2).transpose(1, 2))
            .transpose(1, 2)
            .view(B, -1, h1, w1)
        )

        # 特征拼接与多尺度融合
        fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # 预测并强行上采样恢复到跟输入一模一样的原图大小
        out = self.predictor(self.dropout(fused))
        final_out = F.interpolate(
            out, size=(H, W), mode="bilinear", align_corners=False
        )

        return final_out


def window_partition(x, window_size=8):
    """
    将 [B, H, W, C] 转换为 [B * 个数, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    # 划分成一个个小窗口
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
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """
        x: [B*num_windows, N, C], 其中 N = window_size * window_size
        mask: [num_windows, N, N] 滑动窗口边缘掩码
        """
        B_, N, C = x.shape
        # 计算 Q, K, V
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, head_dim]

        # 算 Attention Score
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 如果存在滑动窗口 Mask，强行斩断跨边缘的不合理注意力
        if mask is not None:
            nW = mask.shape[0]
            # 把 Batch 维度拆开，把 Mask 加到对应的窗口上
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(B_, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)

        # 聚合特征并输出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """标准的 Transformer 前馈网络"""

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2  # 滑动步长为窗口的一半 (4)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

        # 🟢 纯手工最硬核的部分：在初始化时，提前计算好滑动窗口的掩码矩阵 (Buffer)
        self.register_buffer(
            "attn_mask",
            self._create_mask(input_resolution, window_size, self.shift_size),
        )

    def _create_mask(self, res, win_size, shift_size):
        # 创建一个和图像大小一致的辅助矩阵，用来给不同区域打上标签数字
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

        mask_windows = window_partition(
            img_mask, win_size
        )  # [nW, win_size, win_size, 1]
        mask_windows = mask_windows.view(-1, win_size * win_size)
        # 利用广播机制做差，差值不为 0 的地方说明跨越了不该交互的边界
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 将跨边界的位置赋予极大的负值 (-100)，在 Softmax 时权重直接归零
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(self, x):
        # x 形状: [B, H, W, C]
        H, W = self.input_resolution, self.input_resolution
        B, _, _, C = x.shape
        shortcut = x

        # --- 第一阶段：标准局部窗口自注意力 (W-MSA) ---
        x = self.norm1(x)
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=None)  # 常规窗口不需要 mask
        x = window_reverse(
            attn_windows.view(-1, self.window_size, self.window_size, C),
            self.window_size,
            H,
            W,
        )
        x = shortcut + x

        # FFN 层
        x = x + self.mlp(self.norm2(x))

        # --- 第二阶段：移位/滑动窗口自注意力 (SW-MSA) ---
        shortcut = x
        x = self.norm1(x)

        # 核心：使用 torch.roll 算子在空间上进行循环位移
        shifted_x = torch.roll(
            x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
        )

        # 再次切分窗口并注入上面算好的边缘掩码 (attn_mask)
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 还原窗口
        shifted_x = window_reverse(
            attn_windows.view(-1, self.window_size, self.window_size, C),
            self.window_size,
            H,
            W,
        )

        # 核心：计算完毕后，必须沿反方向 roll 回去，把位置对齐
        x = torch.roll(
            shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
        )
        x = shortcut + x

        # FFN 层
        x = x + self.mlp(self.norm2(x))
        return x


class SwinEncoder(nn.Module):
    def __init__(self, in_channels=32, out_channels=128, img_size=128):
        super().__init__()

        # 🟢 区别于标准标准 Swin：为了保护小目标，我们使用 Stride=1 的卷积，坚决不进行下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # 堆叠两个独立的 Swin Block 进行深层空间特征交融
        self.block1 = SwinTransformerBlock(
            dim=out_channels, input_resolution=img_size, num_heads=4, window_size=8
        )
        self.block2 = SwinTransformerBlock(
            dim=out_channels, input_resolution=img_size, num_heads=4, window_size=8
        )

        self.final_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # 输入形状: [B, 32, 128, 128]

        # 1. 提升通道数，保持空间分辨率不变
        x = self.stem(x)  # [B, 128, 128, 128]

        # 2. 转换成 Transformer 内部要求的 [B, H, W, C] 格式
        x = x.permute(0, 2, 3, 1).contiguous()

        # 3. 穿过纯手工打造的 Swin 空间交互网络
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_norm(x)

        # 4. 重新 permute 回 CNN 习惯的 [B, C, H, W] 以无缝对接你的 TransformerShell
        out = x.permute(0, 3, 1, 2).contiguous()
        return out  # 最终输出形状: [B, 128, 128, 128]
