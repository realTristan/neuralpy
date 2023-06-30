import torch
from constants import DEVICE

# Convert the sentence into a tensor
def text_to_tensor(s: str) -> torch.Tensor:
    s_l = list(s.encode("utf-8"))
    return torch.ByteTensor(s_l).to(DEVICE).float()


# Convert all the data into tensors
def to_tensor(data):
    new_data = []
    for i in range(len(data)):
        label_tensor: torch.Tensor = torch.tensor(data[i][1]).to(DEVICE)
        text_tensor: torch.Tensor = text_to_tensor(data[i][0])
        new_data.append((text_tensor, label_tensor))
    return new_data


# Pad the data
def pad(tensors):
    max_len: int = 0
    for tensor in tensors:
        if len(tensor[0]) > max_len:
            max_len = len(tensor[0])

    for i in range(len(tensors)):
        while len(tensors[i][0]) < max_len:
            tmp = tensors[i][0]
            tmp = torch.cat((tensors[i][0], torch.tensor([0]).to(DEVICE)))
            tensors[i] = (tmp, tensors[i][1])
        unsqueezed = tensors[i][0].unsqueeze(0)
        tensors[i] = (unsqueezed, tensors[i][1])
    return tensors
