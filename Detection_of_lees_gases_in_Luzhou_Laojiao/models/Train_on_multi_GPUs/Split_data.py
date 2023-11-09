import numpy as np
import torch


class Masked_dataset():
    """
    生成 masked dataset,
    我们输入的原始光谱是 (number_data, seq_len), 其中seq len就是光谱向量的长度
    我们目前考虑在完整的长度为3321个sampling points组成的2950~3150的光谱范围内
    随机选择两个端点， 截取端点内的光谱，作为训练数据，

    """

    def generate_mask1(self, selected_window_length, total_sequence_length):

        assert selected_window_length < total_sequence_length, "the selected window length should be lower than the total sequence length"
        mask = np.zeros(total_sequence_length)

        start_point = np.random.randint(0, total_sequence_length - selected_window_length)
        end_point = start_point + selected_window_length

        mask[: selected_window_length] = 1
        """
        这里的mask是一个长度为seq len的向量，因为我们目前是截取之后从头放在完整的光谱内，所以从0到截止位置，都是真实的光谱，
        截止位置往后开始，是padding补的0，mask从这里开始全部是0，因此将来在使用mask时候，不会因为光谱本身的0而被分配一个-inf的
        数值在计算attention的时候
        """
        return mask, start_point, end_point

    def generate_mask2(self, total_sequence_length):

        mask = np.zeros(total_sequence_length)

        start_point = 0
        end_point = 0
        # 使用循环确保起点和终点的差大于等于 200
        # 这里200是一个保险，太小了光谱范围太短了，而且通常也不会这么小
        while abs(start_point - end_point) < 500 or end_point <= start_point:
            start_point = np.random.randint(0, total_sequence_length)
            end_point = np.random.randint(0, total_sequence_length)

        mask[: end_point - start_point] = 1

        return mask, start_point, end_point

    def apply_mask(self, dataset, if_fixed_window_length, selected_window_length=None):

        number_data = dataset.shape[0]
        total_data_length = dataset.shape[1]
        checkpoints = np.zeros((number_data, 2))
        mask_list = np.zeros((number_data, total_data_length))
        masked_dataset = np.zeros((number_data, total_data_length))

        if if_fixed_window_length == True:

            for i in range(number_data):
                mask, start, end = self.generate_mask1(selected_window_length, total_data_length)
                mask_list[i] = mask
                masked_dataset[i, :end - start] = dataset[i, start: end]
                checkpoints[i, 0] = start
                checkpoints[i, 1] = end
        else:
            for i in range(number_data):
                mask, start, end = self.generate_mask2(total_data_length)
                mask_list[i] = mask
                masked_dataset[i, :end - start] = dataset[i, start: end]
                checkpoints[i, 0] = start
                checkpoints[i, 1] = end
        """
        mask list就是所有的mask
        chekcpoint记录了在原施光谱中截取的端点的索引，有了这个，将来可以通过checkpoints对应到Nu上了
        self.dataset 就是保存截取之后，并且被padding 0 了的新的数据集，每一行是一个截取的光谱，并且label与原施数据集的label对应

        """
        return mask_list, checkpoints, masked_dataset


if __name__ == "__main__":

    # input_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\input.npy"
    # label_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\label.npy"
    # nu_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\原始数据\波数.npy"

    input_path = r"../../Datasets/original_data/input.npy"
    label_path = r"../../Datasets/original_data/label.npy"
    nu_path = r"../../Datasets/original_data/波数.npy"


    blended_spectra = np.load(input_path)  # (10150, 3321)
    mask_for_full_length = np.ones_like(blended_spectra) # (10150, 3321)
    checkpoint_for_full_length = np.array([0, 3321])
    checkpoint_for_full_length = checkpoint_for_full_length[np.newaxis, :]
    checkpoint_for_full_length = np.repeat(checkpoint_for_full_length, blended_spectra.shape[0], 0)  # [10150, 2]

    label = np.load(label_path)  # (10150, 6)

    nu = np.load(nu_path)  # (3321,)

    repeat_time = 20

    maskset = np.zeros((1, blended_spectra.shape[1]))
    spectraset = np.zeros((1, blended_spectra.shape[1]))
    checkpointset = np.zeros((1, 2))
    masking = Masked_dataset()

    for i in range(repeat_time):

        print(f"[{i+1}/{repeat_time}] iteration finished")

        generated_masks, checkpoint_list, masked_dataset = masking.apply_mask(blended_spectra, False)
        # print(checkpoint_list.shape)

        spectraset = np.vstack((spectraset, masked_dataset))

        maskset = np.vstack((maskset, generated_masks))

        checkpointset = np.vstack((checkpointset, checkpoint_list))

    spectraset = spectraset[1:]
    maskset = maskset[1:]
    checkpointset = checkpointset[1:]

    print(spectraset.shape)
    print(maskset.shape)
    print(checkpointset.shape)

    label_shape = label.shape[1]  # (6)
    new_label = label[np.newaxis, :, :]  # [1, 10150, 6]
    new_label = np.repeat(new_label, repeat_time, 0)  # [10, 10150, 6]
    new_label = new_label.reshape(-1, label_shape)  # [10150* repeat_time, 6]
    new_label[:, 3:5] = new_label[:, 3:5] / 50
    new_label[:, 5] = new_label[:, 5] / 2000
    print(new_label.shape)

    spectraset = np.vstack((spectraset, blended_spectra))
    maskset = np.vstack((maskset, mask_for_full_length))
    checkpointset = np.vstack((checkpointset, checkpoint_for_full_length))

    label[:, 3:5] = label[:, 3:5] / 50
    label[:, 5] = label[:, 5] / 2000
    new_label = np.vstack((new_label, label))

    print(spectraset.shape)
    print(maskset.shape)
    print(checkpointset.shape)
    print(new_label.shape)

    root_path = r"../../Datasets/三组分气体生成的数据集/模拟数据集"
    # root_path = r"F:\学习\Database\Luzhou_Lees_gases_detection\Simulated_dataset"

    save_path1 = root_path + r"/padded_dataset.npy"
    np.save(save_path1, spectraset)

    save_path2 = root_path + r"/masked_dataset_label.npy"
    np.save(save_path2, new_label)

    nu_path = root_path + r"/nu.npy"
    np.save(nu_path, nu)

    mask_path = root_path + r"/mask.npy"
    np.save(mask_path, maskset)

    checkpoints_path = root_path + r"/checkpoints.npy"
    np.save(checkpoints_path, checkpointset)

    print("all data saved!!")
