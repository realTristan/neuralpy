import torch, typing


class Model(torch.nn.Module):
    def __init__(self, size: int = 28, channels: int = 3) -> None:
        super().__init__()
        self.model: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 3),  # 3 input channel, 32 output channels, 3x3 kernel
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),  # 32 input channels, 64 output channels, 3x3 kernel
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3),  # 64 input channels, 64 output channels, 3x3 kernel
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * (size - 6) * (size - 6), 10),  # input features, 10 output features
        )

    # Forward pass
    def forward(self, x) -> typing.Any:
        return self.model(x)

    # Load the model from a file
    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.load_state_dict(torch.load(f))
    
    # Save the model to a file
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            torch.save(self.state_dict(), f)
