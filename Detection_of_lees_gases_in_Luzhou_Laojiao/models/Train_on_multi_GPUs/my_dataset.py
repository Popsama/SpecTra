from PIL import Image
import torch
from torch.utils.data import Dataset


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

    # @staticmethod
    # def collate_fn(batch):
    #     spectra, labels, masks = tuple(zip(*batch))
    #     spectra = torch.as_tensor(spectra)
    #     labels = torch.as_tensor(labels)
    #     masks = torch.as_tensor(masks)
    #     return spectra, labels, masks


if __name__ == "__main__":

    import numpy as np

    save_path1 = r"/home/ser204/zss_204/TLB/remote_server/Detection_of_lees_gases_in_Luzhou_Laojiao/Datasets/三组分气体生成的数据集/模拟数据集/padded_dataset.npy"
    # save_path1 = root_path + r"/padded_dataset.npy"
    spectraset = np.load(save_path1)
    spectraset = spectraset[:, :, np.newaxis]

    save_path2 = r"/home/ser204/zss_204/TLB/remote_server/Detection_of_lees_gases_in_Luzhou_Laojiao/Datasets/三组分气体生成的数据集/模拟数据集/masked_dataset_label.npy"
    label = np.load(save_path2)

    mask_path = r"/home/ser204/zss_204/TLB/remote_server/Detection_of_lees_gases_in_Luzhou_Laojiao/Datasets/三组分气体生成的数据集/模拟数据集/mask.npy"
    maskset = np.load(mask_path)

    train_data_set = MyDataset(spectraset, label, maskset)

    train_loader = torch.utils.data.DataLoader(dataset=train_data_set,
                                               batch_size=128,
                                               shuffle=True)
    train_loader_iter = iter(train_loader)
    x, y, mask = next(train_loader_iter)
    print(x.shape)     # torch.Size([128, 3321, 1])
    print(y.shape)     # torch.Size([128, 6])
    print(mask.shape)  # torch.Size([128, 3321])
