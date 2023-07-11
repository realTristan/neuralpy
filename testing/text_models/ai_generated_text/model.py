import torch
from constants import INPUT_SIZE


# Create a model and dataset and train the padded data
class LSTMModel(torch.nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(INPUT_SIZE, INPUT_SIZE)
        self.linear = torch.nn.Linear(INPUT_SIZE, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
