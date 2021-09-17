import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class CodeLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_encoder_layers=2,
        dropout=0.1,
        is_tadam=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=n_encoder_layers, 
            bidirectional=True,
            dropout=dropout,
        )
        project_in = n_encoder_layers * 2 * d_model
        self.project_layer = nn.Sequential(
            nn.Linear(project_in, d_model), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.d_model = d_model
        self.is_tadam = is_tadam

    def forward(self, x, length=None, tam_embeds=None, **kwargs):
        src_emb = self.embedding(x).transpose(0, 1)
        if tam_embeds is not None and self.is_tadam:  # add TAM as the first token
            tam_embeds = tam_embeds.transpose(0, 1)  # 2xBxD
            num_tam_ts = tam_embeds.size(0)
            src_emb = torch.cat([tam_embeds, src_emb[:-num_tam_ts, :, :]], dim=0)
        src_emb_packed = rnn_utils.pack_padded_sequence(src_emb, length.cpu(), enforce_sorted=False)
        _, (h_n, _) = self.encoder(src_emb_packed)  # TxBxD
        # h_n is n_layers*n_directions x B x d_model
        rep = torch.flatten(h_n.transpose(0, 1), start_dim=1)
        return self.project_layer(rep)
