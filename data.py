import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class Data:
  def __init__(self):
    # Get the training dataset and define the model
    self.training: datasets.MNIST = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    self.model: DataLoader = DataLoader(self.training, batch_size=64, shuffle=True)

    # Get the device
    self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
