import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):

    def __init__(self, input_data, labels, input_masks):
        self.input_data = input_data
        self.labels = labels
        self.input_masks = input_masks

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_item = self.input_data[idx]
        label_item = self.labels[idx]
        mask_item = self.input_masks[idx]

        return input_item, label_item, mask_item


if __name__ == "__main__":
    input_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\input.npy"
    label_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\label.npy"
    nu_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\原始数据\波数.npy"

    blended_spectra = np.load(input_path)
    label = np.load(label_path)
    nu = np.load(nu_path)

    print(blended_spectra.shape)
    print(label.shape)
    print(nu.shape)
