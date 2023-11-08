import torch


def array_to_tensor(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.type(torch.cuda.FloatTensor)
    return tensor