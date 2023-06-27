import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torchvision.transforms import ToTensor
from data import Data
from classifier import Classifier
from trainer import Trainer
import matplotlib.pyplot as plt

# Install PyTorch with CUDA 11.1 (Faster)
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Get the device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the classifier
clf: Classifier = Classifier().to(device)

# Train the model
def train():
    # Get the data
    data: Data = Data()

    # Optimizer
    opt: Adam = Adam(clf.parameters(), lr=0.001)

    # Loss function
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    # Train the model
    Trainer().train(data, clf, opt, loss_fn, device, 10)

    # Save the model
    clf.save("model.pth")

# Test an image
def test(image: str):
    # Load the model
    clf.load("model.pth")

    # Open the image
    image: Image.Image = Image.open(image)
    image_tensor: torch.Tensor = ToTensor()(image).unsqueeze(0).to(device)

    # Get the prediction
    pred: torch.Tensor = torch.argmax(clf(image_tensor))

    # Open the image using matplotlib
    plt.title(f"Prediction: {pred.item()}")
    plt.imshow(image)
    plt.show()


# Run the program
if __name__ == "__main__":
    #train()
    test("images/8.jpg")
