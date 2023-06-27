import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torchvision.transforms import ToTensor
from data import Data
from classifier import Classifier
from trainer import Trainer

# Install PyTorch with CUDA 11.1 (Faster)
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Get the data
data: Data = Data()

# Get the classifier
clf: Classifier = Classifier().to(data.device)

# Create the optimizer
# opt: Adam = Adam(clf.parameters(), lr=0.001)

# Create the loss function
# loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss() #nn.MSELoss()

# Train the model
# Trainer().train(data, clf, opt, loss_fn, 10)

# Save the model
# Trainer.save(clf, "model.pth")

# Load the model
clf.load("model.pth")

# Test an image
image: Image.Image = Image.open("images/2.jpg")
image_tensor: torch.Tensor = ToTensor()(image).unsqueeze(0).to(data.device)
pred: torch.Tensor = torch.argmax(clf(image_tensor))
print(pred)