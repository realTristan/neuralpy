from torch import nn, load

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(1, 32, 3, stride=1), # 1 input channel, 32 output channels, 3x3 kernel, stride 1
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, stride=1), # 32 input channels, 64 output channels, 3x3 kernel, stride 1
      nn.ReLU(),
      nn.Conv2d(64, 64, 3, stride=1), # 64 input channels, 64 output channels, 3x3 kernel, stride 1
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(64*(28-6)*(28-6), 10), # input features, 10 output features
    )
  
  # Forward pass
  def forward(self, x):
    return self.model(x)
  
  # Load the model from a file
  def load(self, path: str):
    with open(path, "rb") as f:
      self.load_state_dict(load(f))
  