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
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

os.environ["WANDB_API_KEY"] = "1d2e465a78a0ca6fbbbb781c9055c095bb90709b"
os.environ["WANDB_MODE"] = "offline"


# 训练函数
def training(config, dataset):

    # 确定我们的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 创建模型
    model = Spec_transformer(config["input_size"],
                             config["output_size"],
                             config["num_layers"],
                             2**config["pre_size"],
                             2**config["attention_size"],
                             2**config["pf_dim"],
                             config["dropout"],
                             2**config["fc1_size"],
                             2**config["fc2_size"],
                             2**config["fc3_size"]).float().to(device)


    # 定义损失函数和优化器
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()

    # dataloader
    batch_size = config["batch_size"]

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / config["training_epoch"])) / 2) * (1 - config["lrf"]) + config["lrf"]
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(config["training_epoch"]):

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

    return mean_loss.item(), mean_cla_loss.item(), mean_reg_loss.item()


def objective(config):

    # hyperopt会给出索引，我们需要使用索引从提供的取值列表中获取具体的值
    config['num_layers'] = [1, 2, 3, 4][config['num_layers']]
    config['pre_size'] = [4, 5, 6, 7, 8, 9][config['pre_size']]
    config['attention_size'] = [4, 5, 6, 7, 8, 9][config['attention_size']]
    config['pf_dim'] = [4, 5, 6, 7, 8, 9][config['pf_dim']]
    config['dropout'] = [0.1, 0.2, 0.3, 0.4][config['dropout']]
    config['fc1_size'] = [7, 8, 9][config['fc1_size']]
    config['fc2_size'] = [7, 8, 9][config['fc2_size']]
    config['fc3_size'] = [7, 8, 9][config['fc3_size']]
    # 'lrf' 和其他固定值的参数不需要使用hp.choice，我们直接设置值
    config['lrf'] = 0.1
    mean_loss, mean_cla_loss, mean_reg_loss = training(config, train_data_set)

    # log the mean loss to wandb website
    wandb.log({"train_loss": mean_loss})
    wandb.log({"mean_cla_loss": mean_cla_loss})
    wandb.log({"mean_reg_loss": mean_reg_loss})
    wandb.log(**config)

    return {'loss': mean_loss, 'status': STATUS_OK}


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

    wandb.init(project="offline-mode",
               dir="/WORK/sunliq_work/TLB/SpecTra/Detection_of_lees_gases_in_Luzhou_Laojiao/models/Train_on_multi_GPUs")

    # 定义超参数搜索空间
    space = {
        # 因为这里是固定值，所以我们不需要对其进行优化
        'training_epoch': 25,
        'batch_size': 128,
        'input_size': 1,
        'output_size': 6,

        # 对于连续值，我们使用hp.uniform定义范围
        'learning_rate': hp.uniform('learning_rate', 0.00001, 0.01),

        # 对于离散值，我们使用hp.choice
        'num_layers': hp.choice('num_layers', [0, 1, 2, 3]),
        'pre_size': hp.choice('pre_size', [0, 1, 2, 3, 4, 5]),
        'attention_size': hp.choice('attention_size', [0, 1, 2, 3, 4, 5]),
        'pf_dim': hp.choice('pf_dim', [0, 1, 2, 3, 4, 5]),
        'dropout': hp.choice('dropout', [0, 1, 2]),
        'fc1_size': hp.choice('fc1_size', [0, 1, 2]),
        'fc2_size': hp.choice('fc2_size', [0, 1, 2]),
        'fc3_size': hp.choice('fc3_size', [0, 1, 2]),
        'lrf': hp.choice('lrf', [0])  # 因为只有一个值，我们也可以直接使用固定值0.1
    }

    # 运行贝叶斯优化
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100
    )

    # 完成W&B记录
    wandb.finish()


    # 放弃使用accelerate 目前没有实现accelerate 与 wandb同步 在多卡上分布式超参数调优




