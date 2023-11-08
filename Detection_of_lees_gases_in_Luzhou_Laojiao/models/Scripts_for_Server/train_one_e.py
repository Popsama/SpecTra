import sys
sys.path.append("../Train_on_multi_GPUs")
from Models import Encoder, Decoder, Spec_transformer
from utils import array_to_tensor
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import tempfile
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataset
# from accelerate import Accelerator
from torch.utils.data import DataLoader


# torch.set_default_dtype(torch.float64)

# load data
# root_path = r"../../Datasets/Triple_gas/simulated_dataset"

root_path = r"../../Datasets/三组分气体生成的数据集/模拟数据集"
save_path1 = root_path + r"/padded_dataset.npy"
spectraset = np.load(save_path1)
spectraset = spectraset[:, :, np.newaxis]

save_path2 = root_path + r"/masked_dataset_label.npy"
label = np.load(save_path2)

mask_path = root_path + r"/mask.npy"
maskset = np.load(mask_path)

spectraset = array_to_tensor(spectraset)
maskset = array_to_tensor(maskset)
label = array_to_tensor(label)

# training config
batch_size = 128
epoch = 2

dataset = MyDataset(spectraset, label, maskset)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Instancing model
num_layers = 1
input_size = 1
pre_size = 32
attention_size = 128
pf_dim = 64
dropout = 0.1

encoder = Encoder(num_layers, input_size, pre_size, attention_size, pf_dim, dropout)
decoder = Decoder(pre_size, attention_size)
spectrans = Spec_transformer(encoder, decoder)

# loss functions
criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.MSELoss()
optimizer = torch.optim.Adam(spectrans.parameters(), lr=0.0001)

# config
loss_list = []
classfication_loss = []
regression_loss = []
device = torch.device("cuda")


spectrans.train()
for e in range(epoch):

    for _, (batch_x, batch_y, batch_x_mask) in enumerate(dataloader):

        weight_list = []

        spectrans = spectrans.to(device)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x_mask = batch_x_mask.to(device)

        optimizer.zero_grad()

        prediction, _ = spectrans(batch_x, batch_x_mask, weight_list)

        loss1 = criterion1(prediction[:, :3], batch_y[:, :3])
        loss2 = criterion2(prediction[:, 3:], batch_y[:, 3:])

        loss = loss1 + loss2

        loss_list.append(float(loss))

        loss.backward()
        optimizer.step()

    print("epoch [{}/{}] \t current loss: {:.4f}".format(e, epoch, loss.item()))

print("training finished")
path = r"../saved_model/saved_sample.pt"
torch.save(spectrans.state_dict(), path)






