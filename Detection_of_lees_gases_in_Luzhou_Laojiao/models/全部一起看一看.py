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

from Enc_module import Encoder, PositionwiseFeedforwardLayer, EncoderLayer
from Dec_module import Decoder, DecoderLayer
from Attn_module import MultiHeadAttentionLayer


#src_mask = [batch size, 1, 1, src len]
def generate_mask(sequence_length, matrix_length):
    mask = torch.zeros(matrix_length)
    mask[:sequence_length] = 1
    return mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(position[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding


if __name__ == "__main__":

    heads_num = 3
    hidden_dim = 12
    seq_len_total = 2000
    batch = 100
    head_dim = 48
    dropout = 0.2
    pf_dim = 64
    num_layers = 2
    decoder_output_shape = 128


    # src mask
    src_mask = generate_mask(890, seq_len_total)
    src_mask = src_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    src_mask = src_mask.repeat(100, 3, 1, 1)

    # original src
    src = torch.rand(batch, seq_len_total, hidden_dim)
    print("src shape :", src.shape)
    # Encoder
    encoder = Encoder(hidden_dim, hidden_dim, num_layers, heads_num, head_dim, pf_dim, dropout, 'cpu', seq_len_total)
    enc_src = encoder(src, src_mask)

    print("enc src shape :", enc_src.shape)


    # Decoder
    decoder = Decoder(decoder_output_shape, hidden_dim, num_layers, heads_num, head_dim, pf_dim, dropout, 'cpu')

    # generate trg
    nu_samples = torch.randn(100, 2000)
    pos_encoding = torch.zeros((1, 2000, 12))
    for i in range(100):
        cache = positional_encoding(nu_samples[i], hidden_dim)
        pos_encoding = torch.cat((pos_encoding, cache), axis=0)

    pos_encoding = pos_encoding[1:, :, :]
    trg = pos_encoding.float()
    print("trg shape :", trg.shape)

    output, _ = decoder(trg, enc_src, src_mask)

    print("output shape :", output.shape)