import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import time
from Enc_module import Encoder3
from Data_loading import MyDataset
from utils import array_to_tensor
import numpy as np


class EncoderLayer(nn.Module):

    def __init__(self,
                 input_dim,
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


####
class Encoder(nn.Module):

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
                                                  dropout).to(self.device(i))
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hid_dim, out_put_dim).to(self.device[-1])

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
        pred = self.fc1(src)

        return pred


#####


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--heads_num', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--head_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--pf_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--seq_len_total', type=int, default=3321)
    parser.add_argument('--output_dim', type=int, default=6)
    opt = parser.parse_args()

    save_path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\padded_dataset.npy"
    spectraset = np.load(save_path1)

    save_path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\masked_dataset_label.npy"
    new_label = np.load(save_path2)

    nu_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\nu.npy"
    nu = np.load(nu_path)

    mask_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\mask.npy"
    maskset = np.load(mask_path)

    checkpoints_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\checkpoints.npy"
    checkpointset = np.load(checkpoints_path)

    print("spectra is with shape of {} ".format(spectraset.shape))
    print("label is with shape of {} ".format(new_label.shape))
    print("wavenumber nu is with shape of {} ".format(nu.shape))
    print("maskset is with shape of {} ".format(maskset.shape))
    print("checkpointset is with shape of {} ".format(checkpointset.shape))


    dataset = MyDataset(spectraset, new_label, maskset)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch, shuffle=True)

    # 分配设备
    device_list = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    device_list2 = ["cpu", "cpu", "cpu", "cpu"]
    # 实例化模型
    encoder = Encoder(opt.hidden_dim, opt.hidden_dim, opt.num_layers, opt.heads_num, opt.head_dim, opt.pf_dim,
                      opt.dropout, opt.seq_len_total, opt.output_dim, device_list2)

    # 跑一个 batch 看看
    for epoch in range(opt.epochs):
        start_time = time.time()  # 记录当前时间
        for i, (src, _, src_mask) in enumerate(dataloader):
            print("src is on {} with shape {}".format(src.device, src.shape))
            print("src_mask is on {} with shape {}".format(src_mask.device, src_mask.shape))

            pred = encoder(src, src_mask)
            print("pred is on device {}".format(pred.device))
            print(pred.shape)
            print("epoch : {} \t round: {}".format(epoch, i))
        end_time  = time.time()  # 记录当前时间
        break
    print(f"finished ! one epoch Time: {end_time - start_time:.2f} seconds'")

