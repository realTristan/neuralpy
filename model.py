from torch import nn, save, load


class Model(nn.Module):
    def __init__(self, size: int = 28) -> None:
        super().__init__()
        self.model: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 32, 3),  # 3 input channel, 32 output channels, 3x3 kernel
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),  # 32 input channels, 64 output channels, 3x3 kernel
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),  # 64 input channels, 64 output channels, 3x3 kernel
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (size - 6) * (size - 6), 10),  # input features, 10 output features
        )

    # Forward pass
    def forward(self, x) -> nn.Sequential:
        return self.model(x)

    # Load the model from a file
    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.load_state_dict(load(f))
    
    # Save the model to a file
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            save(self.state_dict(), f)
