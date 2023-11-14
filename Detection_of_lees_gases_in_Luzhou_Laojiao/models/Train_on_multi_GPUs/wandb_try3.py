import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from utils import array_to_tensor
from my_dataset import MyDataset
import torch.optim as optim
from Models2 import Encoder, Decoder, Spec_transformer
import time
import math
import os
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataset
from accelerate import Accelerator
import wandb


# 训练函数
def training(config, dataset):

    # accelerator
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        # 创建模型
        model = Spec_transformer(config.input_size,
                                 config.output_size,
                                 config.num_layers,
                                 config.pre_size,
                                 config.attention_size,
                                 config.pf_dim,
                                 config.dropout,
                                 config.fc1_size,
                                 config.fc2_size,
                                 config.fc3_size)

    # 定义损失函数和优化器
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()

    # dataloader
    batch_size = config.batch_size

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    if accelerator.is_local_main_process:
        print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / config.training_epoch)) / 2) * (1 - config.lrf) + config.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    accelerator.print(f'device {str(accelerator.device)} is used!')
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model,
                                                                        optimizer,
                                                                        scheduler,
                                                                        train_loader)

    for epoch in range(config.training_epoch):

        model.train()

        mean_loss = torch.zeros(1)
        mean_cla_loss = torch.zeros(1)
        mean_reg_loss = torch.zeros(1)

        optimizer.zero_grad()

        for step, data in enumerate(train_dataloader):
            weight_list = []

            spectra, labels, masks = data

            pred, _ = model(spectra, masks, weight_list)

            loss1 = criterion1(pred[:, :3], labels[:, :3])
            loss2 = criterion2(pred[:, 3:], labels[:, 3:])
            loss = loss1 + 10 * loss2

            # loss.backward()
            accelerator.backward(loss)

            mean_loss = (mean_loss * step + loss.detach().cpu().numpy()) / (step + 1)  # update mean losses
            mean_cla_loss = (mean_cla_loss * step + loss1.detach().cpu().numpy()) / (step + 1)  # update mean losses
            mean_reg_loss = (mean_reg_loss * step + loss2.detach().cpu().numpy()) / (step + 1)  # update mean losses

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        if accelerator.is_local_main_process:

            print("[epoch {}] ".format(epoch))
            # log the mean loss to wandb website
            wandb.log({"train_loss": mean_loss.item()})
            wandb.log({"mean_cla_loss": mean_cla_loss.item()})
            wandb.log({"mean_reg_loss": mean_reg_loss.item()})

        # print logs and save ckpt
        accelerator.wait_for_everyone()

        # unwrapped_model = accelerator.unwrap_model(model)
        # accelerator.save(unwrapped_model, "../saved_model/New_model/saved_model_1.pth")

    wandb.finish()


def train(config=None):
    with wandb.init(config=None):
        config = wandb.config
        training(config, train_data_set)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    # 随机生成数据
    ####################################################################################################################
    # load data
    # root_path = r"../../Datasets/三组分气体生成的数据集/Simulated_dataset"    # Simulated_dataset 大
    root_path = r"../../Datasets/三组分气体生成的数据集/模拟数据集"              # 模拟数据集小

    save_path1 = root_path + r"/padded_dataset.npy"
    spectraset = np.load(save_path1)
    spectraset = spectraset[:, :, np.newaxis]

    save_path2 = root_path + r"/masked_dataset_label.npy"
    label = np.load(save_path2)

    mask_path = root_path + r"/mask.npy"
    maskset = np.load(mask_path)

    ####################################################################################################################

    train_data_set = MyDataset(spectraset, label, maskset)

    ####################################################################################################################

    sweep_config = {
        "method": "bayes",
        "metric": {"goal": "minimize",
                   "name": "train_loss"
                   },
        "parameters": {
            "training_epoch": {
                "value": 2},

            'learning_rate': {
                'distribution': 'uniform',
                'min': 0.00001,
                'max': 0.01
            },

            "batch_size": {"value": 64},

            "input_size": {"value": 1},

            "output_size": {"value": 6},

            "num_layers": {"distribution": "int_uniform",
                           "min": 1,
                           "max": 5},

            "pre_size": {"distribution": "int_uniform",
                         "min": 10,
                         "max": 1024},

            "attention_size": {"distribution": "int_uniform",
                               "min": 10,
                               "max": 1024},

            "pf_dim": {"distribution": "int_uniform",
                       "min": 10,
                       "max": 1024},


            "dropout": {"values": [0.1, 0.2, 0.3, 0.4]},

            "fc1_size": {"distribution": "int_uniform",
                         "min": 10,
                         "max": 1024},

            "fc2_size": {"distribution": "int_uniform",
                         "min": 10,
                         "max": 1024},

            "fc3_size": {"distribution": "int_uniform",
                         "min": 10,
                         "max": 1024},

            "lrf": {"values": [0.1]}
        }
    }



    # 创建wandb sweep
    sweep_id = wandb.sweep(sweep_config)

    # 运行wandb sweep
    wandb.agent(sweep_id, train)


    ## 放弃使用accelerate 目前没有实现accelerate 与 wandb同步 在多卡上分布式超参数调优
