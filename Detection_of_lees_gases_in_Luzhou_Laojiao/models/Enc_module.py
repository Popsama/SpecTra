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


class EncoderLayer(nn.Module):

    def __init__(self,
                 hid_dim,
                 n_heads,
                 head_dim,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, head_dim, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 head_dim,
                 pf_dim,
                 dropout,
                 device,
                 seq_len_total=1000):
        super().__init__()

        self.device = device

        # self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        # self.pos_embedding = nn.Embedding(seq_len_total, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  head_dim,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        # src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src


class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, pf dim]

        return x



class Encoder3(nn.Module):

    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 head_dim,
                 pf_dim,
                 dropout,
                 seq_len,
                 out_put_dim,
                 device_list):
        super().__init__()

        self.device = device_list

        # self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        # self.pos_embedding = nn.Embedding(seq_len_total, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(input_dim, hid_dim,
                                                  n_heads,
                                                  head_dim,
                                                  pf_dim,
                                                  dropout).to(self.device[i])
                                     for i in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        # src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for number, layer in enumerate(self.layers):
            src = layer(src.to(device[number]), src_mask.to(device[number]))

        # src = [batch size, src len, hid dim]

        return src