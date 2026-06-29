"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import json
import os

import torch
from torch.nn import Module
from ...structs import Tensor4D
from .transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerClassifier,
)
from ..interface import Network
from ..embedding import mat2seq, StandardHeightEmbedding, MaskedHeightEmbedding
from ...utils import SegmentationMetrics
from dataclasses import asdict


class TransformerShell(Network):

    def __init__(
        self,
        embedding: StandardHeightEmbedding,
        seq_encoder: TransformerEncoder,
        conv_encoder: torch.nn.Module,
        seq_decoder: TransformerDecoder,
        classifier: TransformerClassifier,
    ):
        super().__init__()
        self.embedding = embedding
        self.seq_encoder = seq_encoder
        self.conv_encoder = conv_encoder
        self.seq_decoder = seq_decoder
        self.classifier: Module = classifier
        # 显式记录 d_model，供后面 view 形状变换使用
        self.d_model = seq_encoder.d_model

    def forward(self, input_mat: Tensor4D, teach_mat=None):
        batch_size, step_len, height, width = input_mat.shape

        # 扁平化输入
        seq_input = mat2seq(input_mat)  # [B * H * W, S]
        seq_mask = seq_input == 0  # [B * H * W, S]

        seq_embedding = self.embedding(seq_input)

        # 将 input_mat 修改为扁平化后的 seq_input 喂给编码器
        pooled_feat, seq_feat = self.seq_encoder(seq_embedding, seq_mask)

        # 转换成 2D 骨干网络需要的 4D 图像特征
        mat_feat = pooled_feat.view(batch_size, height, width, self.d_model)
        mat_feat = mat_feat.permute(0, 3, 1, 2).contiguous()  # [B, d_model, H, W]

        # 横向空间互看
        spatial_feat = self.conv_encoder(mat_feat)  # [B, d_model, H, W]

        # 将空间地图特征精确解构并展平，保证空间索引对齐
        spatial_feat = spatial_feat.permute(0, 2, 3, 1).reshape(
            -1, self.d_model
        )  # [B * H * W, d_model]

        # 纵横特征交汇解码
        refined_feat = self.seq_decoder(spatial_feat, seq_feat, seq_mask)
        logits = self.classifier(refined_feat)  # [B * H * W, S, num_classes]

        # 折叠回五维预测形状
        logits = logits.view(batch_size, height, width, step_len, -1)
        out = logits.permute(0, 4, 3, 1, 2).contiguous()  # [B, num_classes, S, H, W]
        return out

    @torch.no_grad()
    def refer(self, input_mat: Tensor4D):
        valid_len_mat = input_mat > 0
        pred = self.forward(input_mat)
        output = pred.argmax(dim=1) + 1
        return output * valid_len_mat

    def save_checkpoint(self, path: str, best_metrics: SegmentationMetrics):
        """
        将 5 个独立的子模块权重分别保存到指定文件夹下。
        """
        os.makedirs(path, exist_ok=True)

        # 组装待保存的 5 个核心子组件字典
        components = {
            "embedding": self.embedding,
            "seq_encoder": self.seq_encoder,
            "conv_encoder": self.conv_encoder,
            "seq_decoder": self.seq_decoder,
            "classifier": self.classifier,
        }

        for name, sub_module in components.items():
            file_path = os.path.join(path, f"{name}.pth")
            state_dict = sub_module.state_dict()
            torch.save(state_dict, file_path)

            # 核验文件是否真正成功写入且大小正常
            if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
                raise IOError(
                    f"The weight file of submodule [{name}] failed to save or the file is empty!"
                )

        # 保存检查点精度
        metrics_path = os.path.join(path, "best_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(best_metrics), indent=4, ensure_ascii=False))

    def load_checkpoint(self, path: str):
        """
        从指定文件夹中分别读取 5 个子模块的权重，并在真正加载前二次几何核验
        """
        components = {
            "embedding": self.embedding,
            "seq_encoder": self.seq_encoder,
            "conv_encoder": self.conv_encoder,
            "seq_decoder": self.seq_decoder,
            "classifier": self.classifier,
        }

        # 基础物理文件完整性核验
        for name in components.keys():
            file_path = os.path.join(path, f"{name}.pth")
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"\nWeight loading intercepted! Missing key sub component weight file:\n"
                    f"   ➔ Expected file path: {file_path}\n"
                    f"   Please check if the experimental folder or model architecture definition matches."
                )

        # Keys 结构与 Shape 尺寸
        for name, sub_module in components.items():
            file_path = os.path.join(path, f"{name}.pth")

            # 先加载到 CPU
            loaded_state_dict = torch.load(file_path, map_location="cpu")
            current_state_dict = sub_module.state_dict()

            loaded_keys = set(loaded_state_dict.keys())
            current_keys = set(current_state_dict.keys())

            # 计算差异键
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys
            shape_mismatches = []

            # 核验交集 Key 的张量 Shape 是否对齐
            for key in current_keys & loaded_keys:
                if current_state_dict[key].shape != loaded_state_dict[key].shape:
                    shape_mismatches.append(
                        f"      ➔ Attribute '{key}':\n"
                        f"          Runtime model dimension: {list(current_state_dict[key].shape)}\n"
                        f"          File save weight dimension: {list(loaded_state_dict[key].shape)}"
                    )

            # 如果该组件存在任何不一致，立刻抛出详细的崩溃报告，绝不带病运行
            if missing_keys or unexpected_keys or shape_mismatches:
                error_title = (
                    f"\nWeight Dimension Mismatch inside sub-component [{name}]!"
                )
                error_details = []

                if missing_keys:
                    error_details.append(
                        f"   The key that exists in the current model but is missing in the weight file is:\n      {list(missing_keys)}"
                    )
                if unexpected_keys:
                    error_details.append(
                        f"   The weight file contains redundant keys in the current model:\n      {list(unexpected_keys)}"
                    )
                if shape_mismatches:
                    error_details.append(
                        f"   Geometric Dimension Conflict (modified d_model/head/num_classes):\n"
                        + "\n".join(shape_mismatches)
                    )

                raise ValueError(
                    f"{error_title}\n"
                    + "\n\n".join(error_details)
                    + f"\n\nPlease clean up conflicting historical experiment folders or correct the network hyperparameter configuration in 'train. py'."
                )

            sub_module.load_state_dict(loaded_state_dict)


class MaeTransformerShell(TransformerShell):

    def __init__(
        self,
        embedding: MaskedHeightEmbedding,
        seq_encoder: TransformerEncoder,
        conv_encoder: torch.nn.Module,
        seq_decoder: TransformerDecoder,
        classifier: TransformerClassifier,
    ):
        super().__init__(embedding, seq_encoder, conv_encoder, seq_decoder, classifier)

    @torch.no_grad()
    def refer(self, input_mat: Tensor4D):
        pred = self.forward(input_mat)
        return pred

    def save_mae_checkpoint(self, path: str, best_metrics: float):
        """
        将 5 个独立的子模块权重分别保存到指定文件夹下。
        """
        os.makedirs(path, exist_ok=True)

        # 组装待保存的 5 个核心子组件字典
        components = {
            "embedding": self.embedding,
            "seq_encoder": self.seq_encoder,
            "conv_encoder": self.conv_encoder,
            "seq_decoder": self.seq_decoder,
            "classifier": self.classifier,
        }

        for name, sub_module in components.items():
            file_path = os.path.join(path, f"{name}.pth")
            state_dict = sub_module.state_dict()
            torch.save(state_dict, file_path)

            # 核验文件是否真正成功写入且大小正常
            if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
                raise IOError(
                    f"The weight file of submodule [{name}] failed to save or the file is empty!"
                )

        # 保存检查点精度
        metrics_path = os.path.join(path, "best_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"best_metrics": best_metrics}, indent=4, ensure_ascii=False)
            )
