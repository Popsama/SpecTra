import sys
from tqdm import tqdm
import torch
from distributed_utils import reduce_value, is_main_process


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function1, loss_function2):

    model.train()
    mean_loss = torch.zeros(1).to(device)
    mean_cla_loss = torch.zeros(1).to(device)
    mean_reg_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        weight_list = []

        spectra, labels, masks = data

        # spectra = spectra.type(torch.cuda.FloatTensor)
        # labels = labels.type(torch.cuda.FloatTensor)
        # masks = masks.type(torch.cuda.FloatTensor)

        pred, _ = model(spectra.to(device), masks.to(device), weight_list)

        loss1 = loss_function1(pred[:, :3], labels[:, :3].to(device))
        loss2 = loss_function2(pred[:, 3:], labels[:, 3:].to(device))
        loss = loss1 + loss2

        loss.backward()

        loss = reduce_value(loss, average=True)
        loss1 = reduce_value(loss1, average=True)
        loss2 = reduce_value(loss2, average=True)

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        mean_cla_loss = (mean_cla_loss * step + loss1.detach()) / (step + 1)  # update mean losses
        mean_reg_loss = (mean_reg_loss * step + loss2.detach()) / (step + 1)  # update mean losses


        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item(), mean_cla_loss.item(), mean_reg_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item()