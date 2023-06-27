import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torchvision.transforms import ToTensor
from data import Data
from classifier import Classifier
from trainer import Trainer

# Get the data
data: Data = Data()

# Get the classifier
clf: Classifier = Classifier()  # .to("cuda")

# Create the optimizer
opt: Adam = Adam(clf.parameters(), lr=0.001)

# Create the loss function
loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

# Train the model
Trainer().train(data, clf, opt, loss_fn, 5)

# Save the model
Trainer.save(clf, "model.pth")

# Load the model
clf.load(clf, "model.pth")

# Test an image
image: Image.Image = Image.open("images/0.png")
image_tensor: torch.Tensor = ToTensor()(image).unsqueeze(0)
pred: torch.Tensor = torch.argmax(clf(image_tensor))
print(pred)