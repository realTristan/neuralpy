import torch


# Create a dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index][0], self.data[index][1]

    def __len__(self) -> int:
        return len(self.data)