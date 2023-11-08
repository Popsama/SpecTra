import torch
import torch.nn as nn
import torch.optim as optim

import torchtext

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# import spacy
import numpy as np

import random
import math
import time
from Attn_module import MultiHeadAttentionLayer
from Enc_module import PositionwiseFeedforwardLayer


class DecoderLayer(nn.Module):

    def __init__(self,
                 hid_dim,
                 n_heads,
                 head_dim,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, head_dim, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, head_dim, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention


class Decoder(nn.Module):

    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 head_dim,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.device = device

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  head_dim,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_1 = nn.Linear(hid_dim, output_dim)
        self.fc_out = nn.Linear(output_dim, 1)

        self.dropout = nn.Dropout(dropout)

        # self.scale = torch.sqrt(torch.FloatTensor([head_dim])).to(device)

    def forward(self, trg, enc_src, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, n_heads, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        trg = self.fc_1(trg)
        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention