import torch, typing


class Model(torch.nn.Module):
    def __init__(self, size: int = 28, channels: int = 3) -> None:
        super().__init__()
        flc: int = channels * 12 # First layer channels
        slc: int = channels * 24 # Second layer channels
        self.model: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(channels, flc, 3),  # 3 input channel, 32 output channels, 3x3 kernel
            torch.nn.ReLU(),
            torch.nn.Conv2d(flc, slc, 3),  # 32 input channels, 64 output channels, 3x3 kernel
            torch.nn.ReLU(),
            torch.nn.Conv2d(slc, slc, 3),  # 64 input channels, 64 output channels, 3x3 kernel
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(slc * (size - 6) * (size - 6), 10),  # input features, 10 output features
        )

    # Forward pass
    def forward(self, x) -> typing.Any:
        return self.model(x)

    # Load the model from a file
    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.load_state_dict(torch.load(f), strict=False)
    
    # Save the model to a file
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            torch.save(self.state_dict(), f)
