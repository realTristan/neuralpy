from datasets import Datasets
from models import ImageModel
from trainer import Trainer
from image import Image
from constants import DEVICE, MNIST_IMAGE_SIZE
import torch, PIL, matplotlib.pyplot as plt

# Install PyTorch with CUDA 11.1 (Faster)
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Train the model
def train(model, dataset: str, data) -> None:
    # Optimizer
    opt: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loss function
    loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    # Train the model
    Trainer.train(data, model, opt, loss_fn, 10)

    # Save the model
    torch.save(model.state_dict(), dataset)


# Test the mnist dataset
def test_mnist(model, dataset: str, image: str) -> None:
    # Load the model
    model.load_state_dict(torch.load(dataset))

    # Open the image and get the prediction
    image: PIL.Image = PIL.Image.open(image)
    image_tensor: torch.Tensor = Image.to_tensor(image)
    pred: torch.Tensor = torch.argmax(model(image_tensor))

    # Open the image using matplotlib
    plt.title(f"Prediction: {pred.item()}")
    plt.imshow(image)
    plt.show()


# Test the cats and dogs dataset
def test_cats_dogs(model, dataset: str, image: str) -> None:
    # Load the model
    model.load_state_dict(torch.load(dataset))

    # Open the image and get the prediction
    image: PIL.Image = PIL.Image.open(image)
    image_tensor: torch.Tensor = Image.to_tensor(image)
    pred: torch.Tensor = torch.argmax(model(image_tensor))

    # Open the image using matplotlib
    plt.title(f"Prediction: {'Cat' if pred.item() == 0 else 'Dog'}")
    plt.imshow(image)
    plt.show()


# Test the drunk and sober dataset
def test_drunk_sober(model, dataset: str, image: str) -> None:
    # Load the model
    model.load_state_dict(torch.load(dataset))

    # Open the image and get the prediction
    image: PIL.Image = PIL.Image.open(image)
    image_tensor: torch.Tensor = Image.to_tensor(image)
    pred: torch.Tensor = torch.argmax(model(image_tensor))

    # Open the image using matplotlib
    plt.title(f"Prediction: {'Sober' if pred.item() == 0 else 'Drunk'}")
    plt.imshow(image)
    plt.show()


# Run the program
if __name__ == "__main__":
    # Test the cats and dogs dataset
    # model: models.ResNet = models.resnet18().to(DEVICE) # This model is good for large images that need a lot of neurons in the layers
    # data: torch.utils.data.DataLoader = Datasets.fromcsv("cats_dogs.csv", "datasets/cats_dogs")
    # train(model, "models/cats_dogs.pth", data)
    # test_cats_dogs(model, "models/cats_dogs.pth", "datasets/cats_dogs/images/cats/1687885599774338700.jpg")

    # Test the mnist dataset
    model: ImageModel = ImageModel(size=MNIST_IMAGE_SIZE).to(DEVICE) # Our custom model for small images
    data: torch.utils.data.DataLoader = Datasets.mnist()
    train(model, "models/mnist.pth", data)
    test_mnist(model, "models/mnist.pth", "datasets/MNIST/images/9.jpg")

    # Test the drunk and sober dataset
    # model: models.ResNet = models.resnet18().to(DEVICE) # This model is good for large images that need a lot of neurons in the layers
    # data: torch.utils.data.DataLoader = Datasets.fromcsv("drunk_sober.csv", "datasets/drunk_sober")
    # train(model, "models/drunk_sober.pth", data)
    # test_drunk_sober(model, "models/drunk_sober.pth", "datasets/drunk_sober/images/drunk/1687899633180933700.jpg")
