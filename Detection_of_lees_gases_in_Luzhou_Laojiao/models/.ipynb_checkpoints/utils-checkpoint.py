import torch
import numpy as np


class Masked_dataset():

    """
    生成 masked dataset,
    我们输入的原始光谱是 (number_data, seq_len), 其中seq len就是光谱向量的长度
    我们目前考虑在完整的长度为3321个sampling points组成的2950~3150的光谱范围内
    随机选择两个端点， 截取端点内的光谱，作为训练数据，
    为此截取的长度不一，所以通过padding操作，为光谱补零，同时通过mask记录补零的位置，

    """

    def __init__(self, number_data, total_sequence_length):

        self.dataset = np.zeros((number_data, total_sequence_length))

    def generate_mask(self, selected_window_length, total_sequence_length):

        assert selected_window_length < total_sequence_length, "the selected window length should be lower than the total sequence length"

        mask = np.zeros(total_sequence_length)
        checkpoints = np.random.randint(0, total_sequence_length-selected_window_length)
        start_point = checkpoints.item()

        end_point = start_point + selected_window_length

        mask[: end_point] = 1
        """
        这里的mask是一个长度为seq len的向量，因为我们目前是截取之后从头放在完整的光谱内，所以从0到截止位置，都是真实的光谱，
        截止位置往后开始，是padding补的0，mask从这里开始全部是0，因此将来在使用mask时候，不会因为光谱本身的0而被分配一个-inf的
        数值在计算attention的时候
        """
        return mask, start_point, end_point

    def apply_mask(self, dataset):

        number_data = dataset.shape[0]
        total_data_length = dataset.shape[1]
        checkpoints = np.zeros((number_data, 2))
        mask_list = np.zeros((number_data, total_data_length))

        for i in range(number_data):
            mask, start, end = self.generate_mask(total_data_length)
            mask_list[i] = mask
            self.dataset[i, :end-start] = dataset[i, start: end]
            checkpoints[i, 0] = start
            checkpoints[i, 1] = end

        """
        mask list就是所有的mask
        chekcpoint记录了在原施光谱中截取的端点的索引，有了这个，将来可以通过checkpoints对应到Nu上了
        self.dataset 就是保存截取之后，并且被padding 0 了的新的数据集，每一行是一个截取的光谱，并且label与原施数据集的label对应

        """
        return mask_list, checkpoints, self.dataset



# def generate_mask(sequence_length, matrix_length):
#     mask = torch.zeros(matrix_length)
#     mask[:sequence_length] = 1
#     return mask


# def generate_fake_mask(matrix_length):
#
#     checkpoints = torch.randint(0, matrix_length, (1, 2))
#     mask[checkpoints.min().item(): checkpoints.max().item()] = 1
#
#     return mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(position[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding


def array_to_tensor(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.type(torch.cuda.FloatTensor)
    return tensor


if __name__ == "__main__":



    input_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\input.npy"
    label_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\label.npy"
    nu_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\原始数据\波数.npy"

    blended_spectra = np.load(input_path)
    label = np.load(label_path)
    nu = np.load(nu_path)

    repeat_time = 10
    spectraset = np.zeros((1, blended_spectra.shape[1]))
    maskset = np.zeros((1, blended_spectra.shape[1]))
    checkpointset = np.zeros((1, 2))


    for i in range(repeat_time):

        masked_dataset = Masked_dataset(blended_spectra.shape[0], blended_spectra.shape[1])
        generated_masks, checkpoint_list, masked_dataset = masked_dataset.apply_mask(blended_spectra)
        # print(checkpoint_list.shape)

        spectraset = np.vstack((spectraset, masked_dataset))

        maskset = np.vstack((maskset, generated_masks))

        checkpointset = np.vstack((checkpointset, checkpoint_list))

    print(spectraset[1:].shape)
    print(maskset[1:].shape)
    print(checkpointset[1:].shape)


    label_shape = label.shape[1]
    label = label[np.newaxis, :, :]
    label = np.repeat(label, repeat_time, 0)
    new_label = label.reshape(-1, label_shape)
    print(label.shape)

    save_path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\padded_dataset.npy"
    np.save(save_path1, spectraset)

    save_path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\masked_dataset_label.npy"
    np.save(save_path2, new_label)

    nu_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\nu.npy"
    np.save(nu_path, nu)

    mask_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\mask.npy"
    np.save(mask_path, maskset)

    checkpoints_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\Detection_of_lees_gases_in_Luzhou_Laojiao\Datasets\三组分气体生成的数据集\checkpoints.npy"
    np.save(checkpoints_path, checkpointset)



