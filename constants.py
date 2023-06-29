import torch


CHANNELS: int = 3
EPOCHS: int = 10
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MNIST_IMAGE_SIZE: int = 28
