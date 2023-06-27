import torch
from PIL import Image
from torch import nn, load, save
from torch.optim import Adam
from torchvision.transforms import ToTensor
from data import Data
from model import Model
from trainer import Trainer
import matplotlib.pyplot as plt
import torchvision

# Install PyTorch with CUDA 11.1 (Faster)
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Get the device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
# model: Model = Model(size=28).to(device)
model: torchvision.models.ResNet = torchvision.models.resnet18(pretrained=True).to(
    device
)

# Train the model
def train(model_name: str, csv: str = None, path: str = None) -> None:
    # Get the data
    data: Data = Data(csv, path)

    # Optimizer
    opt: Adam = Adam(model.parameters(), lr=0.001)

    # Loss function
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    # Train the model
    Trainer().train(data, model, opt, loss_fn, device, 10)

    # Save the model
    torch.save(model.state_dict(), model_name)


# Test an image
def test(model_name: str, image: str, model) -> None:
    # Load the model
    with open(model_name, "rb") as f:
        model.load_state_dict(load(f))

    # Open the image
    image: Image = Image.open(image)
    image_tensor: torch.Tensor = ToTensor()(image).unsqueeze(0).to(device)

    # Get the prediction
    pred: torch.Tensor = torch.argmax(model(image_tensor))

    # Open the image using matplotlib
    plt.title(f"Prediction: {pred.item()}")
    plt.imshow(image)
    plt.show()


# Test an image (Custom dataset)
def test_custom(model_name: str, image: str, model) -> None:
    # Load the model
    with open(model_name, "rb") as f:
        model.load_state_dict(load(f))

    # Open the image
    image: Image = Image.open(image)
    image_tensor: torch.Tensor = ToTensor()(image).unsqueeze(0).to(device)

    # Update channels
    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    # Get the prediction
    pred: torch.Tensor = torch.argmax(model(image_tensor))

    # Open the image using matplotlib
    plt.title(f"Prediction: {'Cat' if pred.item() == 0 else 'Dog'}")
    plt.imshow(image)
    plt.show()


# Run the program
if __name__ == "__main__":
    # train("dogs_cats_model.pth", csv="dogs_cats.csv", path="custom_dataset")
    test_custom(
        "dogs_cats_model.pth",
        "custom_dataset/images/cats/1687885599774338700.jpg",
        model,
    )
