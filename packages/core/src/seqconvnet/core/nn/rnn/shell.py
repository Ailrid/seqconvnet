"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from dataclasses import asdict
import json
import os

import torch
from ..interface import Network
from .rnn import RnnEncoder, RnnDecoder, RnnClassifier
from ...structs import SegmentationMetrics, Tensor4D
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
    def refer(self, input_mat: Tensor4D):
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

            # 减回去
            output[:, i : i + 1, :, :] = current_layer_label - 1

        return output

    def save_checkpoint(self, path: str, best_metrics: SegmentationMetrics):
        """
        将 5 个独立的子模块权重分别保存到指定文件夹下。
        """
        os.makedirs(path, exist_ok=True)

        # 组装待保存的 5 个核心子组件字典
        components = {
            "embedding_encoder": self.embedding_encoder,
            "embedding_decoder": self.embedding_decoder,
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
            "embedding_encoder": self.embedding_encoder,
            "embedding_decoder": self.embedding_decoder,
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


class RnnChunkShell(RnnShell):

    def __init__(
        self,
        embedding_encoder: torch.nn.Module,
        embedding_decoder: torch.nn.Module,
        seq_encoder: RnnEncoder,
        conv_encoder: torch.nn.Module,
        seq_decoder: RnnDecoder,
        classifier: RnnClassifier,
        chunk_size: int = 1024,  # 新增参数：控制每一批处理的序列数量，可根据显存动态调整（如 512, 1024, 2048）
    ):
        super().__init__(
            embedding_encoder,
            embedding_decoder,
            seq_encoder,
            conv_encoder,
            seq_decoder,
            classifier,
        )

        # 保存分批大小
        self.chunk_size = chunk_size

    def forward(self, input_mat: Tensor4D, teach_mat: Tensor4D):
        """训练前向传播"""
        batch_size, step_len, num_rows, num_cols = input_mat.shape
        total_seqs = batch_size * num_rows * num_cols  # 你的 16384 个序列总数

        # pooled_feat_mat -> [B, num_layers * hidden_size, H, W]
        pooled_feat_mat = self.encode(input_mat)

        # pooled_feat_seq -> [num_layers, B * H * W, hidden_size]
        pooled_feat_seq = pooled_feat_mat.view(
            batch_size, self.num_layers, self.hidden_size, num_rows, num_cols
        )
        pooled_feat_seq = pooled_feat_seq.permute(1, 0, 3, 4, 2).contiguous()
        pooled_feat_seq = pooled_feat_seq.view(
            self.num_layers, total_seqs, self.hidden_size
        )

        # 将标签矩阵打扁、查表、调整时序轴
        teach_seq = mat2seq(teach_mat).to(torch.int64)  # [B * H * W, S]
        embedding_teach = self.embedding_decoder(teach_seq).permute(
            1, 0, 2
        )  # [S, B * H * W, embed_size]

        # 分批（Chunking）通过 Decoder 和 Classifier
        logits_chunks = []
        for i in range(0, total_seqs, self.chunk_size):
            # 切片当前分批的特征与标签
            emb_teach_chunk = embedding_teach[:, i : i + self.chunk_size, :]
            feat_seq_chunk = pooled_feat_seq[:, i : i + self.chunk_size, :]

            # output_chunk -> [S, chunk_size, hidden_size]
            output_chunk = self.seq_decoder(emb_teach_chunk, feat_seq_chunk)

            # logits_chunk -> [chunk_size, S, num_classes]
            logits_chunk = self.classifier(output_chunk)
            logits_chunks.append(logits_chunk)

        # 在第 0 维（B * H * W 维度）将所有分批的 logits 拼接回来
        logits = torch.cat(logits_chunks, dim=0)
        # -----------------------------------------------------------------

        # 用 logits 还原空间对齐
        out_mat = logits.view(
            batch_size, num_rows, num_cols, step_len, self.num_classes
        )
        out_mat = out_mat.permute(0, 4, 3, 1, 2).contiguous()
        return out_mat  # [B, num_classes, S, H, W]

    def encode(self, input_mat: Tensor4D):
        """混合编码管道"""
        batch_size, _, num_rows, num_cols = input_mat.shape
        total_seqs = batch_size * num_rows * num_cols

        seq_mat = mat2seq(input_mat).to(torch.int64)
        # [S, B * H * W, embed_size]
        embedding_seq = self.embedding_encoder(seq_mat).permute(1, 0, 2)

        # 分批（Chunking）通过 Encoder
        state_seq_chunks = []
        for i in range(0, total_seqs, self.chunk_size):
            emb_seq_chunk = embedding_seq[:, i : i + self.chunk_size, :]

            # chunk_state -> [num_layers, chunk_size, hidden_size]
            chunk_state = self.seq_encoder(emb_seq_chunk)
            state_seq_chunks.append(chunk_state)

        # 在第 1 维（B * H * W 维度）将所有分批的隐状态拼接回来
        state_seq = torch.cat(state_seq_chunks, dim=1)
        # -----------------------------------------------------------------

        # 转换并组装空间 4D 特征
        state_mat = state_seq.view(
            self.num_layers, batch_size, num_rows, num_cols, self.hidden_size
        )
        state_mat = state_mat.permute(1, 0, 4, 2, 3).contiguous()
        state_mat = state_mat.view(
            batch_size, self.num_layers * self.hidden_size, num_rows, num_cols
        )

        return state_mat + self.conv_encoder(state_mat)
