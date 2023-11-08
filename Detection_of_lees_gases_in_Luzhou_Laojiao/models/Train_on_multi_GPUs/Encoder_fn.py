import torch
import torch.nn as nn
import torch.nn.functional as F
# import spacy
import numpy as np
from Attn import Attention, PositionwiseFeedforwardLayer


class EncoderLayer(nn.Module):

    def __init__(self,
                 input_size,
                 pre_size,
                 attention_size,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(pre_size)
        self.ff_layer_norm = nn.LayerNorm(pre_size)
        self.self_attention = Attention(pre_size, attention_size)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(pre_size,
                                                                     pf_dim,
                                                                     dropout)
        self.pre_process = nn.Linear(input_size, pre_size)
        self.dropout = nn.Dropout(dropout)
        # self.attention_weights = attention_weights

    def forward(self, src, src_mask, attention_weights):
        # src = [batch size, src_len, input_dim]
        # src_mask = [batch size, src len]

        # src = [batch, seq len, pre_size]
        src = self.pre_process(src)

        # self attention
        _src, weights = self.self_attention(src, src_mask)
        attention_weights.append(weights)
        # src = [batch, seq_len, input_dim]

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src, attention_weights


class Encoder(nn.Module):

    # 目前先用一层的encoder试试

    def __init__(self,
                 n_layers,
                 input_size,
                 pre_size,
                 attention_size,
                 pf_dim,
                 dropout,
                 device_list):
        super().__init__()

        self.device = device_list

        self.layers = nn.ModuleList([EncoderLayer(input_size,
                                                  pre_size,
                                                  attention_size,
                                                  pf_dim,
                                                  dropout).to(self.device[i])
                                     for i in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask, attention_weights):
        for number, layer in enumerate(self.layers):
            x, attention_weights = layer(x.to(self.device[number]), x_mask.to(self.device[number]),
                                         attention_weights)

        return x, attention_weights


if __name__ == "__main__":

    a = torch.randn(10, 3321, 1)
    a_mask = torch.zeros(10, 3321)
    attention_weights = []

    enc_layer = EncoderLayer(1, 12, 24, 32, 0.1)
    context, weights = enc_layer(a, a_mask, attention_weights)
    print(context.shape)  # torch.Size([10, 3321, 12])
    print(weights[0].shape)  # torch.Size([10, 3321, 1])

    encoder = Encoder(1, 1, 12, 24, 32, 0.1, ["cpu"])
    encoded_a, weights = encoder(a, a_mask, attention_weights)
    print(encoded_a.shape)    # torch.Size([10, 3321, 12])
    print(weights[0].shape)   # torch.Size([10, 3321, 1])


