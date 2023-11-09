from Models2 import Encoder, Decoder, Spec_transformer
from utils import array_to_tensor
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import time
# import matplotlib.pyplot as plt
import numpy as np
import os
import math
import tempfile
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataset
from accelerate import Accelerator
import wandb


def training_loop(args, dataset, model, epochs, learning_rate, lrf,
                  num_layers, pre_size, attention_size,
                  pf_dim, dropout, fc1_size, fc2_size, fc3_size,
                  criterion1, criterion2):
    ####################################################################################################################

    # accelerator
    accelerator = Accelerator()

    # initialize the model
    model = model(args.input_size,
                  args.output_size,
                  args.num_layers,
                  args.pre_size,
                  args.attention_size,
                  args.pf_dim,
                  args.dropout,
                  args.fc1_size,
                  args.fc2_size,
                  args.fc3_size)

    # dataloader
    batch_size = args.batch_size

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    if accelerator.is_local_main_process:
        print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    accelerator.print(f'device {str(accelerator.device)} is used!')
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model,
                                                                        optimizer,
                                                                        scheduler,
                                                                        train_loader)

    loss_list = []
    classification_loss = []
    regression_loss = []

    training_start_time = time.time()

    for epoch in range(epochs):

        start_time = time.time()  # 记录当前时间

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
            loss = loss1 + 10*loss2

            # loss.backward()
            accelerator.backward(loss)

            mean_loss = (mean_loss * step + loss.detach().cpu().numpy()) / (step + 1)  # update mean losses
            mean_cla_loss = (mean_cla_loss * step + loss1.detach().cpu().numpy()) / (step + 1)  # update mean losses
            mean_reg_loss = (mean_reg_loss * step + loss2.detach().cpu().numpy()) / (step + 1)  # update mean losses


            optimizer.step()
            optimizer.zero_grad()

        # log the mean loss to wandb website
        wandb.log({"train_loss": mean_loss.item()})

        loss_list.append(mean_loss.item())
        classification_loss.append(mean_cla_loss.item())
        regression_loss.append(mean_reg_loss.item())

        end_time = time.time()  # 记录当前时间

        scheduler.step()
        if accelerator.is_local_main_process:
            print("[epoch {}] mean loss: {:.4f}    classification loss: {:.4f}   "
                  "regression loss: {:.4f}  one epoch Time: {:.2f} min ".format(epoch,
                                                                                mean_loss.item(),
                                                                                mean_cla_loss.item(),
                                                                                mean_reg_loss.item(),
                                                                                (end_time-start_time)/60))

        # print logs and save ckpt
        accelerator.wait_for_everyone()

        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model, "../saved_model/New_model/saved_model_1.pth")

    training_end_time = time.time()
    # 计算训练用时（秒数）
    training_duration = training_end_time - training_start_time
    # 将秒数转换为“天 时 分 秒”格式
    days, remainder = divmod(training_duration, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    # 格式化输出
    formatted_duration = f"{int(days)} day {int(hours)} hour {int(minutes)} min {int(seconds)} sec"

    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model, "../saved_model/New_model/saved_model_1.pth")

    np.save("../saved_model/New_model/total_loss.npy", np.array(loss_list))
    np.save("../saved_model/New_model/cla_loss.npy", np.array(classification_loss))
    np.save("../saved_model/New_model/reg_loss.npy", np.array(regression_loss))

    if accelerator.is_local_main_process:
        print("Training time:", formatted_duration, "used")


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    ####################################################################################################################
    # load data
    # root_path = r"../../Datasets/三组分气体生成的数据集/Simulated_dataset"
    root_path = r"../../Datasets/三组分气体生成的数据集/模拟数据集"

    save_path1 = root_path + r"/padded_dataset.npy"
    spectraset = np.load(save_path1)
    spectraset = spectraset[:, :, np.newaxis]

    save_path2 = root_path + r"/masked_dataset_label.npy"
    label = np.load(save_path2)

    mask_path = root_path + r"/mask.npy"
    maskset = np.load(mask_path)

    ####################################################################################################################


    train_data_set = MyDataset(spectraset, label, maskset)

    sweep_config = {
        "name": "training_parameter tuning",
        "method": "bayes",
        "metric": {"goal": "minimize",
                   "name": "train_loss"
                   },
        "parameters": {"training_epoch": {"value": 5000},
                       "learning_rate": {"values": [0.01, 0.001, 0.0001, 0.00001]},
                       "batch_size": {"value": 256},
                       "num_layers": {"distribution": "int_uniform",
                                            "min": 1,
                                            "max": 5
                                            },

                       "pre_size": {"distribution": "int_uniform",
                                         "min": 10,
                                         "max": 1024
                                         },
                       "attention_size": {"distribution": "int_uniform",
                                         "min": 10,
                                         "max": 1024
                                         },
                       "pf_dim": {"distribution": "int_uniform",
                                         "min": 10,
                                         "max": 1024
                                         },
                       "dropout": {"values": [0.1, 0.2, 0.3, 0.4]
                                  },

                       "fc1_size": {"distribution": "int_uniform",
                                    "min": 10,
                                    "max": 1024
                                    },
                       "fc2_size": {"distribution": "int_uniform",
                                    "min": 10,
                                    "max": 1024
                                    },
                       "fc3_size": {"distribution": "int_uniform",
                                    "min": 10,
                                    "max": 1024
                                    },
                       "lrf": {"values": [0.1]}
                       }
    }


    # wandb.login()
    wandb.init(project="SpecTra Hyper-parameter tuning")
    sweep_id = wandb.sweep(sweep_config, project="SpecTra Hyper-parameter tuning")

    # Instancing model
    input_size = 1
    output_size = 6



    # loss functions
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()

    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    opt = parser.parse_args()

    print(opt)

    def train(config=None):
        with wandb.init(config=None):
            config = wandb.config
            training_loop(opt, train_data_set, spectrans, criterion1, criterion2)




