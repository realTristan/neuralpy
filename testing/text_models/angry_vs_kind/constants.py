import torch


INPUT_SIZE: int = 216
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS: int = 1000