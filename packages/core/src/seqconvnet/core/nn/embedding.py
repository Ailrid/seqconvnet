import torch
from torch import nn


def mat2seq(x):
    """将空间体素矩阵打扁为序列格式: [B, S, H, W] -> [B * H * W, S]"""
    num_steps = x.shape[1]
    return x.permute(0, 2, 3, 1).reshape(-1, num_steps)


class StandardHeightEmbedding(nn.Module):
    def __init__(self, max_z=128, embed_size=32):
        super().__init__()
        # 长度为 max_z + 2 (包含 0~max_z 真实高度以及 Padding, EOS)
        # 当 max_z=128 时，长度为 130 (有效索引 0 到 129)
        max_len = max_z + 2

        P = torch.zeros((max_len, embed_size))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, embed_size, 2, dtype=torch.float32) / embed_size
        )
        P[:, 0::2] = torch.sin(X)
        P[:, 1::2] = torch.cos(X)

        # 冻结的静态物理表
        self.register_buffer("static_table", P)

    def forward(self, x):
        # 查表
        return self.static_table[x]  # type: ignore


class HybridHeightEmbedding(StandardHeightEmbedding):
    def __init__(self, max_z=128, embed_size=32):
        super().__init__()
        max_len = max_z + 2

        # 1. 🛠️ 依然保留你引以为傲的静态物理表（提供坚实的物理和泛化底座）
        P = torch.zeros((max_len, embed_size))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, embed_size, 2, dtype=torch.float32) / embed_size
        )
        P[:, 0::2] = torch.sin(X)
        P[:, 1::2] = torch.cos(X)
        self.register_buffer("static_table", P)

        # 2. 🛠️ 引入一个可学习的“残差修正表”
        self.learnable_residual = nn.Embedding(max_len, embed_size)

        # 3. 🛠️ 极其关键：使用极小的标准差初始化（比如 0.001），或者全 0 初始化
        # 这样在训练刚启动和前几轮，它几乎等于0，模型完全依赖静态表
        nn.init.normal_(self.learnable_residual.weight, mean=0.0, std=0.001)

    def forward(self, x):
        # 静态大底座 + 动态小修正
        return self.static_table[x] + self.learnable_residual(x)  # type: ignore


class MaskedHeightEmbedding(StandardHeightEmbedding):
    def __init__(self, embed_size, max_z=128):
        super().__init__()
        # 【修正】改名叫 mask_id，避免和下面的 nn.Parameter 冲突
        self.mask_id = max_z + 2
        self.eos = max_z + 1

        # 大表总长度设为 max_z + 3 = 131
        max_len = max_z + 3
        P = torch.zeros((max_len, embed_size))

        # 填充正余弦几何规律
        X = torch.arange(max_z + 2, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, embed_size, 2, dtype=torch.float32) / embed_size
        )
        P[: max_z + 2, 0::2] = torch.sin(X)
        P[: max_z + 2, 1::2] = torch.cos(X)
        self.register_buffer("static_table", P)

        # 可学习的掩码特征占位符
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_size) * 0.02)

    def forward(self, x):
        # 查表，拿到 [B*H*W, S, d_model]
        feat = self.static_table[x]  # type: ignore

        # 【修正】使用 self.mask_id 进行精准布尔匹配
        mask_indices = (x == self.mask_id).unsqueeze(-1)  # [B*H*W, S, 1]

        # 把 MASK_ID 查出来的全 0 向量，替换为可学习的 mask_token
        out_feat = torch.where(mask_indices, self.mask_token, feat)
        return out_feat
