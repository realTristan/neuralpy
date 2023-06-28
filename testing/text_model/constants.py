import torch


INPUT_SIZE: int = 152
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS: int = 100