from Models import Encoder, Decoder, Spec_transformer
from utils import array_to_tensor
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Process

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
from distributed_utils import init_distributed_mode, dist, cleanup
from train_eval_utils import train_one_epoch


def main(args, dataset, model, criterion1, criterion2):

    ####################################################################################################################
    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息
        print(args)
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)

    # 实例化模型
    model = model.to(device)

    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")

    # 需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    # pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loss_list = []
    classification_loss = []
    regression_loss = []

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss, cla_loss, reg_loss = train_one_epoch(model=model,
                                                        optimizer=optimizer,
                                                        data_loader=train_loader,
                                                        epoch=epoch,
                                                        loss_function1=criterion1,
                                                        loss_function2=criterion2)

        loss_list.append(float(mean_loss))
        classification_loss.append(float(cla_loss))
        regression_loss.append(float(reg_loss))

        scheduler.step()

        if rank == 0:
            print("[epoch {}] mean loss: {:.4f}".format(epoch, mean_loss))
            torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()

    ####################################################################################################################

    # plotting the training process
    loss_list = np.array(loss_list)
    Loss = []
    for i in range(args.epochs):
        Loss.append(np.mean(loss_list[i * 80: (i + 1) * 80]))
    Loss = np.array(Loss)

    plt.figure()
    plt.grid()
    plt.yscale("log")
    plt.plot(Loss)
    # plt.show()

    Loss = []
    for i in range(args.epochs):
        Loss.append(np.mean(classification_loss[i * 80: (i + 1) * 80]))
    Loss = np.array(Loss)

    plt.figure()
    plt.style.use('seaborn-paper')
    plt.grid()
    plt.yscale("log")
    plt.plot(Loss)
    # plt.show()

    Loss = []
    for i in range(args.epochs):
        Loss.append(np.mean(regression_loss[i * 80: (i + 1) * 80]))
    Loss = np.array(Loss)

    plt.figure()
    plt.style.use('seaborn-paper')
    plt.grid()
    plt.yscale("log")
    plt.plot(Loss)
    plt.show()


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    # load data
    # root_path = r"/zss_204/TLB/remote_server/Detection_of_lees_gases_in_Luzhou_Laojiao/Datasets/三组分气体生成的数据集/模拟数据集"
    save_path1 = r"/home/ser204/zss_204/TLB/remote_server/Detection_of_lees_gases_in_Luzhou_Laojiao/Datasets/三组分气体生成的数据集/模拟数据集/padded_dataset.npy"
    # save_path1 = root_path + r"/padded_dataset.npy"
    spectraset = np.load(save_path1)
    spectraset = spectraset[:, :, np.newaxis]

    save_path2 = r"/home/ser204/zss_204/TLB/remote_server/Detection_of_lees_gases_in_Luzhou_Laojiao/Datasets/三组分气体生成的数据集/模拟数据集/masked_dataset_label.npy"
    label = np.load(save_path2)

    mask_path = r"/home/ser204/zss_204/TLB/remote_server/Detection_of_lees_gases_in_Luzhou_Laojiao/Datasets/三组分气体生成的数据集/模拟数据集/mask.npy"
    maskset = np.load(mask_path)

    spectraset = array_to_tensor(spectraset)
    maskset = array_to_tensor(maskset)
    label = array_to_tensor(label)

    train_data_set = MyDataset(spectraset, label, maskset)

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


    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')

    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()

    main(opt, train_data_set, spectrans, criterion1, criterion2)