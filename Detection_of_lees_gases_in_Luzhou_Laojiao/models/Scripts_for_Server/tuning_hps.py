import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from utils import array_to_tensor
import torch.optim as optim
from Models2 import Encoder, Decoder, Spec_transformer
import math
import os
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataset
import wandb
from tqdm import tqdm

# os.environ["WANDB_API_KEY"] = "1d2e465a78a0ca6fbbbb781c9055c095bb90709b"
# os.environ["WANDB_MODE"] = "offline"


# 训练函数
def training(config, dataset):

    # 确定我们的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 创建模型
    model = Spec_transformer(config.input_size,
                             config.output_size,
                             config.num_layers,
                             2**config.pre_size,
                             2**config.attention_size,
                             2**config.pf_dim,
                             config.dropout,
                             2**config.fc1_size,
                             2**config.fc2_size,
                             2**config.fc3_size).float().to(device)

    # # 查看模型参数的类型
    # for param in model.parameters():
    #     print("parameters dtype: {}".format(param.dtype))

    # 定义损失函数和优化器
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()

    # dataloader
    batch_size = config.batch_size

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / config.training_epoch)) / 2) * (1 - config.lrf) + config.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(config.training_epoch):

        model.train()

        mean_loss = torch.zeros(1)
        mean_cla_loss = torch.zeros(1)
        mean_reg_loss = torch.zeros(1)

        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc='Epoch {:03d}'.format(epoch+1), leave=True, disable=False)

        for step, data in enumerate(progress_bar):

            weight_list = []
            spectra, labels, masks = data

            spectra = spectra.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            pred, _ = model(spectra, masks, weight_list)

            loss1 = criterion1(pred[:, :3], labels[:, :3])
            loss2 = criterion2(pred[:, 3:], labels[:, 3:])
            loss = loss1 + 10 * loss2

            # loss.backward()
            loss.backward()

            mean_loss = (mean_loss * step + loss.detach().cpu().numpy()) / (step + 1)  # update mean losses
            mean_cla_loss = (mean_cla_loss * step + loss1.detach().cpu().numpy()) / (step + 1)  # update mean losses
            mean_reg_loss = (mean_reg_loss * step + loss2.detach().cpu().numpy()) / (step + 1)  # update mean losses

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'mean_loss': loss.item()})

        scheduler.step()

        # log the mean loss to wandb website
        wandb.log({"train_loss": mean_loss.item()})
        wandb.log({"mean_cla_loss": mean_cla_loss.item()})
        wandb.log({"mean_reg_loss": mean_reg_loss.item()})


    # wandb.finish()


def train(config=None):
    with wandb.init(config=None, project="Remote_server hyper-parameter tuning",
                    dir="/WORK/sunliq_work/TLB/SpecTra/Detection_of_lees_gases_in_Luzhou_Laojiao/models/Train_on_multi_GPUs/wandb"):
        config = wandb.config
        training(config, train_data_set)


if __name__ == "__main__":

    """
    现在尝试了，num_layer = 4, batch_size = 128, 其他参数设置为2^9，Tesla V100 30GB GPU可以承担。所以上限就是这些数值。
    在这个基础上 epoch=200试试 sweep parameters
    """

    # torch.set_default_dtype(torch.float64)

    # 随机生成数据
    ####################################################################################################################
    # load data
    # root_path = r"../../Datasets/三组分气体生成的数据集/Simulated_dataset"    # Simulated_dataset 大
    # root_path = r"../../Datasets/三组分气体生成的数据集/模拟数据集"              # 模拟数据集小

    root_path = r"/WORK/sunliq_work/TLB/SpecTra/Detection_of_lees_gases_in_Luzhou_Laojiao/Datasets/Triple_gas/Small"
    # Small 更 小

    save_path1 = root_path + r"/padded_dataset.npy"
    spectraset = np.load(save_path1)
    spectraset = spectraset[:, :, np.newaxis]

    save_path2 = root_path + r"/masked_dataset_label.npy"
    label = np.load(save_path2)

    mask_path = root_path + r"/mask.npy"
    maskset = np.load(mask_path)

    ####################################################################################################################

    spectraset = torch.from_numpy(spectraset).float()
    label = torch.from_numpy(label).float()
    maskset = torch.from_numpy(maskset).float()

    train_data_set = MyDataset(spectraset, label, maskset)

    ####################################################################################################################

    sweep_config = {

        "method": "bayes",

        "metric": {"goal": "minimize",
                   "name": "train_loss"
                   },

        "parameters": {
            "training_epoch": {
                "value": 50},

            'learning_rate': {
                'distribution': 'uniform',
                'min': 0.00001,
                'max': 0.01
            },

            "batch_size": {"value": 128},

            "input_size": {"value": 1},

            "output_size": {"value": 6},

            "num_layers": {"distribution": "int_uniform",
                           "min": 1,
                           "max": 4},

            "pre_size": {"distribution": "int_uniform",
                         "min": 4,
                         "max": 9},

            "attention_size": {"distribution": "int_uniform",
                               "min": 4,
                               "max": 9},

            "pf_dim": {"distribution": "int_uniform",
                       "min": 4,
                       "max": 9},

            "dropout": {"values": [0.1, 0.2, 0.3, 0.4]},

            "fc1_size": {"distribution": "int_uniform",
                         "min": 7,
                         "max": 9},

            "fc2_size": {"distribution": "int_uniform",
                         "min": 7,
                         "max": 9},

            "fc3_size": {"distribution": "int_uniform",
                         "min": 7,
                         "max": 9},

            "lrf": {"values": [0.1]}
        }
    }

    # 创建 wandb sweep
    sweep_id = wandb.sweep(sweep_config, project="Remote_server hyper-parameter tuning")

    # 运行wandb sweep
    wandb.agent(sweep_id, train)

    # 放弃使用accelerate 目前没有实现accelerate 与 wandb同步 在多卡上分布式超参数调优




