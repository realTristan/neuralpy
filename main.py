from data import Data
from model import Model
from trainer import Trainer
from image import Image
from torchvision import models
import torch, PIL, matplotlib.pyplot as plt

# Install PyTorch with CUDA 11.1 (Faster)
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Get the device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
def train(model, dataset: str, csv: str = None, path: str = None) -> None:
    # Get the data
    data: Data = Data(csv, path)

    # Optimizer
    opt: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loss function
    loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    # Train the model
    Trainer().train(data, model, opt, loss_fn, device, 10)

    # Save the model
    torch.save(model.state_dict(), dataset)


# Test an image
def test(model, dataset: str, image: str) -> None:
    # Load the model
    with open(dataset, "rb") as f:
        model.load_state_dict(torch.load(f))

    # Open the image and get the prediction
    image: PIL.Image = PIL.Image.open(image)
    image_tensor: torch.Tensor = Image.to_tensor(image, device)
    pred: torch.Tensor = torch.argmax(model(image_tensor))

    # Open the image using matplotlib
    plt.title(f"Prediction: {pred.item()}")
    plt.imshow(image)
    plt.show()


# Test an image (Custom dataset)
def test_custom(model, dataset: str, image: str) -> None:
    # Load the model
    with open(dataset, "rb") as f:
        model.load_state_dict(torch.load(f))

    # Open the image and get the prediction
    image: PIL.Image = PIL.Image.open(image)
    image_tensor: torch.Tensor = Image.to_tensor(image, device)
    pred: torch.Tensor = torch.argmax(model(image_tensor))

    # Open the image using matplotlib
    plt.title(f"Prediction: {'Cat' if pred.item() == 0 else 'Dog'}")
    plt.imshow(image)
    plt.show()


# Run the program
if __name__ == "__main__":
    # Initialize the model
    model: models.ResNet = models.resnet18().to(device)
    # train(model, "models/dogs_cats_model.pth", csv="dogs_cats.csv", path="custom_dataset")
    test_custom(model, "models/dogs_cats_model.pth", "custom_dataset/images/cats/1687885599774338700.jpg") 
    # custom_dataset/images/cats/1687885599774338700.jpg
    # custom_dataset/images/dogs/1687885598581945400.jpg

    # Initialize the model
    # model: Model = Model(size=28).to(device)
    # train(model, "models/numbers.pth")
    # test(model, "models/numbers.pth", "images/0.jpg")
