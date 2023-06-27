import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class Data:
    def __init__(self):
        # Get the training dataset and define the model
        self.mnist: datasets.MNIST = datasets.MNIST(
            root="data", train=True, download=True, transform=ToTensor()
        )
        self.model: DataLoader = DataLoader(self.mnist, batch_size=64, shuffle=True)
