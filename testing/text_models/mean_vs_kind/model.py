import torch
from constants import INPUT_SIZE


# Create a model and dataset and train the padded data
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=128, out_channels=1, kernel_size=2
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.model(x)
        

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
