import torch, typing


class Model(torch.nn.Module):
    def __init__(self, size: int = 28, channels: int = 3) -> None:
        super().__init__()
        hlc: int = channels * 12 # Hidden layer channels
        olc: int = channels * 24 # Output layer channels
        self.model: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hlc, 3),  # Input layer. (Input channels, Output channels, 3x3 kernel)
            torch.nn.ReLU(),
            torch.nn.Conv2d(hlc, olc, 3),  # Hidden layer. (Input channels, Output channels, 3x3 kernel)
            torch.nn.ReLU(),
            torch.nn.Conv2d(olc, olc, 3),  # Output layer. (Input channels, Output channels, 3x3 kernel)
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(olc * (size - 6) * (size - 6), 10),  # Input features, 10 output features
            # olc (output layers) * (matrix size) * (matrix size)
            # The matrix size becomes smaller as the image goes through the layers because of matrix multiplication
            # We subsctract 6 because we have 3 layers with a 3x3 kernel. 3 * 2 = 6 (every layer 2 matrix columns are expelled)
            # The (matrix size) * (matrix size) is the size of the image (28x28 for example)
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
