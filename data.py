from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class Data:
  # Get the dataset
  def __init__(self):
    self.training = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    self.model = DataLoader(self.training, batch_size=64, shuffle=True)
