import torch
import torch.nn as nn
import torch.nn.functional as F
# import spacy
import numpy as np
from Attn import Aggregation_Attention, PositionwiseFeedforwardLayer
from Encoder_fn import EncoderLayer, Encoder


class DecoderLayer(nn.Module):

    def __init__(self,
                 pre_size,
                 attention_size
                 ):
        super().__init__()

        self.ag_attn_layer = Aggregation_Attention(pre_size, attention_size)

        self.fc1 = nn.Linear(in_features=pre_size, out_features=196)
        nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=196, out_features=774)
        nn.init.xavier_normal_(self.fc2.weight)

        self.fc3 = nn.Linear(in_features=774, out_features=211)
        nn.init.xavier_normal_(self.fc3.weight)

        self.fc4 = nn.Linear(in_features=211, out_features=6)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x, x_mask):
        # x = [batch, seq_len, pre_size]
        # x_mask = [batch, seq_len]
        x, attention_weight = self.ag_attn_layer(x, x_mask)

        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.sigmoid(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.sigmoid(x)
        # x = self.dropout1(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)

        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x, attention_weight


class Decoder(nn.Module):

    def __init__(self,
                 pre_size,
                 attention_size,
                 device_list
                 ):
        super().__init__()

        self.device = device_list

        self.layers = DecoderLayer(pre_size, attention_size)

    def forward(self, x, x_mask):
        x, attention_weights = self.layers(x.to(self.device), x_mask.to(self.device))

        return x, attention_weights


if __name__ == "__main__":

    a = torch.randn(10, 3321, 1)
    a_mask = torch.zeros(10, 3321)
    attention_weights = []

    # enc_layer = EncoderLayer(1, 12, 24, 32, 0.1)
    # context, weights = enc_layer(a, a_mask, attention_weights)
    # print(context.shape)  # torch.Size([10, 3321, 12])
    # print(weights[0].shape)  # torch.Size([10, 3321, 1])

    encoder = Encoder(1, 1, 12, 24, 32, 0.1, ["cpu"])
    encoded_a, weights = encoder(a, a_mask, attention_weights)
    print(encoded_a.shape)    # torch.Size([10, 3321, 12])
    print(len(weights))
    print(weights[0].shape)   # torch.Size([10, 3321, 1])

    # decoder_layer = DecoderLayer(12, 32)
    # decoded_a, weight = decoder_layer(encoded_a, a_mask)
    # weights.append(weight)
    # print(decoded_a.shape)
    # print(len(weights))
    # print(weights[1].shape)

    decoder = Decoder(12, 32, "cpu")
    decoded_a, weight = decoder(encoded_a, a_mask)
    weights.append(weight)
    print(decoded_a.shape)
    print(len(weights))
    print(weights[1].shape)
