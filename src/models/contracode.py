"""
Transformer architecture based on https://arxiv.org/abs/2007.04973.
https://github.com/parasj/contracode/blob/master/representjs/models/transformer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from src.models.adapter import AdapterLayer
from src.models.film import FilmShiftScaleLayer


class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, dropout=0.1, max_len=9000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

    def _load_from_state_dict(self, *args):
        print("PositionalEncoding: doing nothing on call to _load_from_state_dict")


class CodeTransformerEncoder(nn.Module):
    
    def __init__(
            self,
            n_tokens,
            d_model=768,
            n_head=8,
            n_encoder_layers=6,
            d_ff=2048,
            dropout=0.1,
            activation="relu",
            norm=True,
            pad_id=None,
            is_tam=False,
            is_tadam=False,
            is_adapter=False,
        ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9000)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_dim = d_model
        encoder_layer = TransformerEncoderLayer(encoder_dim, n_head, d_ff, dropout, activation, 768*2, is_tadam, is_adapter)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            tam_embeds=None,
            tadam_or_adapter_embeds=None,
        ):
        src_emb = self.embedding(input_ids).transpose(0, 1) * math.sqrt(self.config["d_model"])
        if tam_embeds is not None:  # add TAM as the first token
            tam_embeds = tam_embeds.transpose(0, 1)  # 2xBxD
            num_tam_ts = tam_embeds.size(0)
            src_emb = torch.cat([tam_embeds, src_emb[:-num_tam_ts, :, :]], dim=0)
        src_emb = self.pos_encoder(src_emb)  # TxBxD
        src_key_padding_mask = (1 - attention_mask).bool()
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask, task_embeds=tadam_or_adapter_embeds)  # TxBxD
        out = out.permute(1, 0, 2).contiguous()  # BxTxD
        return (out,)


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, task_embeds=None):
        """Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
        Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_key_padding_mask, task_embeds=task_embeds)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    
    TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            d_task=768*2,
            is_tadam=False,
            is_adapter=False,
        ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        if is_tadam:
            self.film1 = FilmShiftScaleLayer(d_task, d_model, 64)
            self.film2 = FilmShiftScaleLayer(d_task, d_model, 64)
        elif is_adapter:
            self.adapter1 = AdapterLayer(d_model + d_task, d_model, 64)
            self.adapter2 = AdapterLayer(d_model + d_task, d_model, 64)

        self.is_tadam, self.is_adapter = is_tadam, is_adapter

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)
    
    def condition_layer_one(self, src, task_embeds):
        if self.is_tadam:
            gamma, beta = self.film1(task_embeds)
            src = src * gamma.expand_as(src) + beta.expand_as(src)
        elif self.is_adapter:
            task_embeds = task_embeds.unsqueeze(0)
            task_embeds = task_embeds.repeat(src.size(0), 1, 1)
            src = self.adapter1(src, task_embeds)
        return src

    def condition_layer_two(self, src, task_embeds):
        if self.is_tadam:
            gamma, beta = self.film2(task_embeds)
            src = src * gamma.expand_as(src) + beta.expand_as(src)
        elif self.is_adapter:
            task_embeds = task_embeds.unsqueeze(0)
            task_embeds = task_embeds.repeat(src.size(0), 1, 1)
            src = self.adapter2(src, task_embeds)
        return src

    def forward(self, src, src_key_padding_mask=None, task_embeds=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.condition_layer_one(src2, task_embeds)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.condition_layer_two(src2, task_embeds)

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
