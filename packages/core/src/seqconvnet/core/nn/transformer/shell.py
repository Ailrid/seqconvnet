"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from torch.nn import Module
from ...structs import Tensor4D
from .transformer import TransformerEncoder, TransformerDecoder, TransformerClassifier
import torch
from ..interface import Network


class TransformerShell(Network):

    def __init__(
        self,
        seq_encoder: TransformerEncoder,
        conv_encoder: torch.nn.Module,
        seq_decoder: TransformerDecoder,
        classifier: TransformerClassifier,
    ):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.conv_encoder = conv_encoder
        self.seq_decoder = seq_decoder
        self.classifier: Module = classifier

    def forward(self, input_mat: Tensor4D, valid_len_mat: Tensor4D):
        seq_state = self.seq_encoder(
            input_mat, valid_len_mat
        )  # [batch_size*num_rows*num_cols,d_model]
        conv_state = self.conv_encoder(
            seq_state[0]
        )  # [batch_size*num_rows*num_cols,d_model]
        decoder_state = self.seq_decoder(seq_state[1], conv_state, valid_len_mat)
        return self.classifier(decoder_state, valid_len_mat)

    @torch.no_grad()
    def refer(self, input_mat: Tensor4D, valid_len_mat: Tensor4D):
        pred = self.forward(input_mat, valid_len_mat)
        output = pred.argmax(dim=1) + 1
        return output * valid_len_mat
